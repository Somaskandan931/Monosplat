"""
colab/train.py — Entry point for MonoSplat training (Colab + Desktop).

Uses only modules that exist in this project:
  src/utils/         — config_loader, colmap_utils, io_utils
  src/dataset/       — loader
  src/preprocessing/ — normalize_scene
  src/reconstruction/— gaussian_model, trainer
  core/reconstruction/— checkpoint_manager
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import shutil

import numpy as np
import torch

# ── sys.path setup ────────────────────────────────────────────────────────────
# _REPO_ROOT is the project root (parent of colab/)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC       = _REPO_ROOT / "src"

for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports from src/ ────────────────────────────────────────────────────────
from utils.config_loader import load_config
from utils.colmap_utils   import load_colmap_model, get_sparse_point_cloud
from utils.io_utils       import save_ply, save_splat
from dataset.loader       import ColmapDataset
from reconstruction.gaussian_model import GaussianModel
from reconstruction.trainer         import Trainer

# Core checkpoint manager may not exist in this repo (no core/ directory).
# Use a minimal local checkpoint manager instead.
from typing import Optional, Dict, Any

class CheckpointManager:
    """Minimal checkpoint helper for Colab + Desktop.

    Responsibilities:
      - auto-resume from an existing checkpoint if --resume not provided
      - optional mirroring of checkpoints to a drive directory

    File format: torch.save(dict) containing at least:
      iteration, model, optimizer, scaler
    """

    def __init__(self, checkpoints_dir: Path, mirror_dir: Optional[Path] = None):
        self.checkpoints_dir = checkpoints_dir
        self.mirror_dir = mirror_dir

    def auto_resume_path(self, resume_path: Optional[str]) -> Optional[Path]:
        if resume_path:
            p = Path(resume_path)
            return p if p.exists() else None
        latest = self.latest_checkpoint_path()
        return latest

    def latest_checkpoint_path(self) -> Optional[Path]:
        if not self.checkpoints_dir.exists():
            return None
        ckpts = sorted(self.checkpoints_dir.glob("checkpoint_*.ckpt"), key=lambda p: p.stat().st_mtime)
        return ckpts[-1] if ckpts else None

    def write_resume_report(self, report_path: Path, resume_path: Optional[Path]) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"resume_path": str(resume_path) if resume_path else None}
        report_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    def mirror_checkpoint(self, ckpt_path: str | Path, drive_checkpoint_dir: str | Path) -> None:
        src = Path(ckpt_path)
        if not src.exists():
            return
        dst_root = Path(drive_checkpoint_dir)
        dst_root.mkdir(parents=True, exist_ok=True)
        dst = dst_root / src.name
        try:
            __import__("shutil").copy2(src, dst)
        except Exception:
            # Best-effort mirror only
            pass


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Safety constants ──────────────────────────────────────────────────────────
# Raised from 50_000 → 150_000 in BLUR-FIX-1 to improve initialization density.
# Lowered to 100_000 here because simple_knn.distCUDA2 corrupts CUDA memory
# (cudaErrorIllegalAddress / SIGABRT) when N exceeds ~65k, and gaussian_model.py
# now routes large N through _dist_gpu_chunked instead. 100k is a safe ceiling
# that keeps initialization density high while staying well inside the chunked
# path. Reconstruction quality is not materially affected: 3DGS densification
# adds Gaussians from iter 500 onward regardless of seed density.
MAX_INIT_GAUSSIANS: int = 100_000


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a MonoSplat scene.")
    p.add_argument("--sparse",  dest="source_path", required=True,
                   help="Path to COLMAP sparse_text folder")
    p.add_argument("--frames",  dest="image_dir",   required=True,
                   help="Path to extracted frames folder")
    p.add_argument("--output",  dest="model_path",  required=True,
                   help="Output model directory")
    p.add_argument("--config",  default=str(_REPO_ROOT / "configs" / "config.yaml"),
                   help="YAML config path")
    p.add_argument("--resume",  default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip scene normalization (not recommended)")
    p.add_argument("--iterations", type=int, default=None,
                   help="Override config training.iterations")
    return p.parse_args()


# ── GPU-tier preset overrides (from Colab notebook env var) ──────────────────

def _apply_env_overrides(cfg: dict) -> None:
    """
    Read MONOSPLAT_EXTRA_TRAIN_ARGS set by the Colab GPU-tier preset cell
    and apply the values into cfg["training"].

    Format: --training.key value  (e.g. --training.max_gaussians 80000)
    """
    raw = os.environ.get("MONOSPLAT_EXTRA_TRAIN_ARGS", "").strip()
    if not raw:
        return

    log.info("[train] Applying env overrides: %s", raw)
    tokens = raw.split()
    i = 0
    while i < len(tokens) - 1:
        flag  = tokens[i]
        value = tokens[i + 1]
        i += 2

        if not flag.startswith("--training."):
            continue
        key = flag[len("--training."):]

        if key in cfg.get("training", {}):
            orig = cfg["training"][key]
            try:
                if isinstance(orig, bool):
                    coerced = value.lower() in ("1", "true", "yes")
                elif isinstance(orig, int):
                    coerced = int(value)
                elif isinstance(orig, float):
                    coerced = float(value)
                else:
                    coerced = value
                cfg["training"][key] = coerced
                log.info("[train]   training.%s = %s (was %s)", key, coerced, orig)
            except (ValueError, TypeError) as exc:
                log.warning("[train]   Could not apply override %s %s: %s", flag, value, exc)
        else:
            # New key — best-effort int/float coercion
            try:
                coerced = int(value)
            except ValueError:
                try:
                    coerced = float(value)
                except ValueError:
                    coerced = value
            cfg.setdefault("training", {})[key] = coerced
            log.info("[train]   training.%s = %s (new key)", key, coerced)


# ── Scene helpers ─────────────────────────────────────────────────────────────

def compute_cameras_extent(dataset: ColmapDataset) -> float:
    """Estimate scene scale from camera positions (median pairwise distance)."""
    centers = dataset.get_all_camera_centers()  # (N, 3)
    if len(centers) < 2:
        return 1.0
    diff  = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    upper = dists[np.triu_indices(len(centers), k=1)]
    return max(float(np.median(upper)) if len(upper) > 0 else 1.0, 0.1)


def subsample_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    seed: int = 42,
) -> tuple:
    """Random-subsample point cloud to at most max_points with fixed seed."""
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    log.warning(
        "Point cloud subsampled: %d → %d points (%d%% kept). "
        "Reason: prevents OOM during initialise_from_pcd.",
        n, max_points, int(max_points / n * 100),
    )
    return xyz[idx], rgb[idx]


# ── Drive persistence ─────────────────────────────────────────────────────────

def _prepare_drive_persistence(cfg: dict) -> Path | None:
    runtime = cfg.setdefault("runtime", {})
    if not runtime.get("drive_persistence", True):
        return None
    drive_root = Path(os.environ.get(
        "MONOSPLAT_DRIVE_ROOT", runtime.get("drive_root", "")
    ))
    if not str(drive_root) or str(drive_root) == ".":
        return None
    if not drive_root.parent.exists():
        return None
    for subdir in ["datasets", "checkpoints", "exports"]:
        (drive_root / subdir).mkdir(parents=True, exist_ok=True)
    return drive_root


def _copy_exports_to_drive(export_dir: Path, drive_export_dir: Path) -> None:
    drive_export_dir.mkdir(parents=True, exist_ok=True)
    for artifact in export_dir.glob("*"):
        if artifact.is_file():
            shutil.copy2(artifact, drive_export_dir / artifact.name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg["source_path"] = args.source_path
    cfg["image_dir"]   = args.image_dir
    cfg["model_path"]  = args.model_path

    # CLI --iterations override (applied first; env overrides win if both set)
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations

    # Apply GPU-tier overrides from Colab notebook env var
    _apply_env_overrides(cfg)

    # Restore CLI override if env tried to clobber it
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations

    drive_root = _prepare_drive_persistence(cfg)

    # ── Output + checkpoint directories ──────────────────────────────────────
    model_dir = Path(args.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(
        model_dir / "checkpoints",
        drive_root / "checkpoints" if drive_root else None,
    )
    resume_path = checkpoint_manager.auto_resume_path(args.resume)
    checkpoint_manager.write_resume_report(
        model_dir / "resume_report.json", resume_path
    )
    if resume_path:
        log.info("[checkpoint] Resuming from: %s", resume_path)
    else:
        log.info("[checkpoint] Fresh start (no checkpoint found).")

    # ── Config audit log ──────────────────────────────────────────────────────
    t = cfg["training"]
    log.info("=" * 60)
    log.info("EFFECTIVE TRAINING CONFIG (after all overrides)")
    log.info("  iterations             : %s", t["iterations"])
    log.info("  max_gaussians          : %s", t.get("max_gaussians", "N/A"))
    log.info("  densify_from/until     : %s → %s",
             t["densify_from_iter"], t["densify_until_iter"])
    log.info("  densification_interval : %s", t["densification_interval"])
    log.info("  densify_grad_threshold : %s", t["densify_grad_threshold"])
    log.info("  lambda_dssim           : %s", t.get("lambda_dssim", 0.2))
    log.info("  lambda_lpips           : %s", t.get("lambda_lpips", 0.05))
    log.info("  sh_degree              : %s", cfg["model"]["sh_degree"])
    log.info("  renderer.max_gaussians : %s",
             cfg["renderer"].get("max_gaussians", "N/A"))
    log.info("  position_lr_init       : %s", t.get("position_lr_init"))
    log.info("  position_lr_max_steps  : %s", t.get("position_lr_max_steps"))
    log.info("=" * 60)

    # ── Load COLMAP model ─────────────────────────────────────────────────────
    log.info("Loading COLMAP sparse model from: %s", args.source_path)
    cameras_colmap, images_colmap, points3D = load_colmap_model(args.source_path)

    # ── Scene normalization ───────────────────────────────────────────────────
    if not args.no_normalize:
        log.info("Normalizing scene (translate to origin, scale to unit sphere) …")
        from preprocessing.normalize_scene import normalize_scene, scene_stats
        pre  = scene_stats(images_colmap, points3D)
        log.info("  Before: extent=%s", pre.get("camera_extent", "?"))
        images_colmap, points3D, norm_info = normalize_scene(images_colmap, points3D)
        post = scene_stats(images_colmap, points3D)
        log.info(
            "  After:  extent=%s  scale=%.6f  centroid=%s",
            post.get("camera_extent", "?"),
            norm_info["scale"],
            norm_info["centroid"].round(3).tolist(),
        )
    else:
        log.warning(
            "Scene normalization DISABLED (--no_normalize). "
            "Gaussian explosion may occur on off-origin scenes."
        )
        norm_info = {"centroid": np.zeros(3), "scale": 1.0, "applied": False}

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = ColmapDataset(
        frames_dir=args.image_dir,
        sparse_dir=args.source_path,
    )
    # Inject normalized images so Camera.from_colmap uses correct tvec values
    dataset.images = images_colmap
    train_cameras  = dataset.views

    # ── Large-dataset guard ───────────────────────────────────────────────────
    num_images  = len(train_cameras)
    if num_images > 250:
        config_iters = cfg["training"]["iterations"]
        safe_cap     = min(config_iters, 25000)
        if config_iters > safe_cap:
            log.warning(
                "Large dataset (%d images). Capping iterations: %d → %d.",
                num_images, config_iters, safe_cap,
            )
            cfg["training"]["iterations"] = safe_cap

    log.info(
        "Dataset: %d images | Iterations: %d | Max Gaussians: %s",
        num_images,
        cfg["training"]["iterations"],
        cfg["training"].get("max_gaussians", "N/A"),
    )

    # ── Scene extent ──────────────────────────────────────────────────────────
    cameras_extent = compute_cameras_extent(dataset)
    # FP-2 (foggy preview fix): after normalize_scene the scene fits inside a
    # ~0.047-radius sphere.  compute_cameras_extent() then returns ~0.1 because
    # all camera centres are clustered near the origin.  Passing 0.1 to
    # densify_and_prune() makes the screen-size pruning threshold 10× too tight,
    # killing most Gaussians.  After normalization the scene IS unit-scale by
    # construction, so cameras_extent < 1.0 is always an underestimate — clamp.
    cameras_extent_raw = cameras_extent
    cameras_extent = max(cameras_extent, 1.0)
    log.info(
        "cameras_extent raw=%.4f → clamped=%.4f  "
        "(clamped to 1.0 so densify_and_prune screen-size thresholds are correct; "
        "raw < 1.0 means normalize_scene compressed cameras — this is expected)",
        cameras_extent_raw, cameras_extent,
    )
    dataset.cameras_extent = cameras_extent

    # ── Point cloud → Gaussian init ───────────────────────────────────────────
    xyz_np, rgb_np = get_sparse_point_cloud(points3D)
    xyz_np, rgb_np = subsample_point_cloud(xyz_np, rgb_np, MAX_INIT_GAUSSIANS)

    xyz = torch.from_numpy(xyz_np).float()
    rgb = torch.from_numpy(rgb_np).float()

    log.info("Initialising %d Gaussians from sparse point cloud.", len(xyz))

    model = GaussianModel(sh_degree=cfg["model"]["sh_degree"])
    # FP-1 (foggy preview fix): pass cameras_extent as spatial_lr_scale so
    # initialise_from_pcd computes the initial Gaussian size ceiling relative
    # to the actual scene radius rather than an absolute 1.0 world-unit.
    # With cameras_extent=1.0 (post-normalization floor), each Gaussian starts
    # at diameter ≈ 0.1 world-units — correct for a unit-sphere scene.
    model.initialise_from_pcd(xyz, rgb, spatial_lr_scale=cameras_extent)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(cfg=cfg, model=model, scene=dataset)

    if resume_path:
        trainer._setup_optimizer()
        trainer.resume_from_checkpoint(str(resume_path))

    trainer.train()

    # ── Export ────────────────────────────────────────────────────────────────
    export_dir = model_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    gaussians = model.get_state()
    save_ply(str(export_dir / "final.ply"),     gaussians)
    save_splat(str(export_dir / "final.splat"), gaussians)

    if drive_root:
        _copy_exports_to_drive(
            export_dir,
            drive_root / "exports",
        )

    log.info("Exported final artifacts to: %s", export_dir)
    log.info("Training complete.")


if __name__ == "__main__":
    main()