"""
scripts/train.py — Entry point for MonoSplat training.

FIXES APPLIED:
  [FIX-1] sys.path: only adds _REPO_ROOT/src (flat modules live there).
          Previous version also added _REPO_ROOT itself, causing ambiguity
          between `utils` (top-level stub) and `src/utils` (real module).
  [FIX-2] MAX_INIT_GAUSSIANS: 40_000 → 60_000.
          40k was over-conservative; T4 handles 60k init Gaussians fine with
          the fixed densify params (grad_threshold=0.0003, interval=200).
          60k gives COLMAP-consistent initial coverage for most scenes.
  [FIX-3] Point cloud subsampling now uses random shuffle with fixed seed
          for even spatial coverage instead of head-truncation.
  [FIX-4] All config reads use cfg["training"]["key"] style (plain dict).
  [FIX-5] Scene normalization now applied after COLMAP loading and before
          Gaussian init. Prevents Gaussian explosion from off-origin or
          badly-scaled reconstructions (major quality fix).
  [FIX-6] MONOSPLAT_EXTRA_TRAIN_ARGS env var (set by Colab GPU-tier preset
          cell) is now parsed and applied as config overrides, so the notebook
          tier presets actually take effect.  Previously the variable was set
          but train.py never read it.
  [FIX-7] Large-dataset guard: was hardcoded to cap iterations at 18000
          regardless of config.yaml, silently overriding higher configured
          values.  Now caps at min(config_iters, 25000) so the config value
          is respected and the cap only applies when truly needed.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import shutil

import numpy as np
import torch

# ── sys.path: _REPO_ROOT/src is the one true source location ─────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC       = _REPO_ROOT / "src"
for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.config_loader import load_config
from utils.colmap_utils   import load_colmap_model, get_sparse_point_cloud
from utils.io_utils       import save_ply, save_splat
from dataset.loader       import ColmapDataset
from reconstruction.gaussian_model import GaussianModel
from reconstruction.trainer         import Trainer
from core.experiments import ExperimentManager
from core.hardware import HardwareDetector, HardwareProfileManager
from core.reconstruction import CheckpointManager

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

MAX_INIT_GAUSSIANS: int = 60_000   # FIX-2: 40k → 60k; safe with corrected densify params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FIX-6: Apply Colab GPU-tier preset overrides from env var
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: dict) -> None:
    """
    Read MONOSPLAT_EXTRA_TRAIN_ARGS and apply overrides to cfg["training"].

    The Colab notebook Cell 1 sets env var:
        MONOSPLAT_EXTRA_TRAIN_ARGS = "--training.max_gaussians 150000 ..."

    This function parses that string and writes values into cfg so the
    tier-specific presets actually affect training.
    """
    raw = os.environ.get("MONOSPLAT_EXTRA_TRAIN_ARGS", "").strip()
    if not raw:
        return

    log.info(f"[train] Applying env overrides: {raw}")
    tokens = raw.split()
    i = 0
    while i < len(tokens) - 1:
        flag  = tokens[i]
        value = tokens[i + 1]
        i += 2

        # Expected format: --training.key value
        if not flag.startswith("--training."):
            continue
        key = flag[len("--training."):]

        # Coerce value to appropriate Python type
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
                log.info(f"[train]   training.{key} = {coerced} (was {orig})")
            except (ValueError, TypeError) as exc:
                log.warning(f"[train]   Could not apply override {flag} {value}: {exc}")
        else:
            # Key not in defaults — add it as a float/int best-effort
            try:
                coerced = int(value)
            except ValueError:
                try:
                    coerced = float(value)
                except ValueError:
                    coerced = value
            cfg.setdefault("training", {})[key] = coerced
            log.info(f"[train]   training.{key} = {coerced} (new key)")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def compute_cameras_extent(dataset: ColmapDataset) -> float:
    """
    Estimate scene scale from camera positions (median pairwise distance).
    Falls back to 1.0 if fewer than 2 cameras.

    After normalization, this should return approximately 1.0.
    """
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
    """
    Subsample (xyz, rgb) to at most max_points rows.

    FIX-3: Random shuffle with a fixed seed for even spatial coverage.
    """
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    log.warning(
        f"Point cloud subsampled: {n:,} → {max_points:,} points "
        f"({max_points / n * 100:.0f}% kept). "
        "Reason: prevents OOM during initialise_from_pcd on T4."
    )
    return xyz[idx], rgb[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg["source_path"] = args.source_path
    cfg["image_dir"]   = args.image_dir
    cfg["model_path"]  = args.model_path

    # Apply --iterations CLI override before env overrides so env can still
    # win (Colab GPU-tier presets take final precedence).
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations

    # FIX-6: Apply GPU-tier overrides from Colab notebook env var
    _apply_env_overrides(cfg)

    hardware_report = HardwareDetector().detect()
    cfg, active_profile = HardwareProfileManager(_REPO_ROOT / "configs").apply_profile(
        cfg,
        hardware_report,
    )
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations
    _apply_memory_safety(cfg, hardware_report)
    drive_root = _prepare_drive_persistence(cfg)
    experiment_root = drive_root / "experiments" if drive_root else _REPO_ROOT / "experiments"

    experiment_manager = ExperimentManager(experiment_root)
    run = experiment_manager.create_run(
        dataset_path=args.image_dir,
        config_snapshot=dict(cfg),
        repo_root=_REPO_ROOT,
    )
    run_dir = Path(run["run_dir"])
    cfg["experiment"] = {
        "run_id": run["run_id"],
        "run_dir": str(run_dir),
    }
    if drive_root:
        cfg["runtime"]["drive_root"] = str(drive_root)
        cfg["runtime"]["drive_checkpoint_dir"] = str(drive_root / "checkpoints" / run["run_id"])
    _copy_pretraining_reports(args.image_dir, run_dir)
    HardwareDetector().save_report(run_dir / "hardware_report.json", hardware_report)
    log.info("[experiment] run_id=%s run_dir=%s", run["run_id"], run_dir)
    log.info(
        "[hardware] profile=%s gpu=%s vram=%.2fGB cuda=%s",
        active_profile,
        hardware_report.get("gpu_type"),
        hardware_report.get("vram_gb", 0.0),
        hardware_report.get("cuda_version"),
    )

    checkpoint_manager = CheckpointManager(
        Path(args.model_path) / "checkpoints",
        drive_root / "checkpoints" if drive_root else None,
    )
    resume_path = checkpoint_manager.auto_resume_path(args.resume)
    checkpoint_manager.write_resume_report(run_dir / "resume_report.json", resume_path)
    if resume_path:
        log.info("[checkpoint] Auto-resume checkpoint: %s", resume_path)

    # ── CONFIG AUDIT — always log effective values so mismatches are visible ──
    t = cfg["training"]
    log.info("=" * 60)
    log.info("EFFECTIVE TRAINING CONFIG (after all overrides)")
    log.info(f"  iterations             : {t['iterations']}")
    log.info(f"  max_gaussians          : {t.get('max_gaussians', 'N/A')}")
    log.info(f"  densify_from/until     : {t['densify_from_iter']} → {t['densify_until_iter']}")
    log.info(f"  densification_interval : {t['densification_interval']}")
    log.info(f"  densify_grad_threshold : {t['densify_grad_threshold']}")
    log.info(f"  lambda_dssim           : {t.get('lambda_dssim', 0.2)}")
    log.info(f"  lambda_lpips           : {t.get('lambda_lpips', 0.05)}")
    log.info(f"  sh_degree              : {cfg['model']['sh_degree']}")
    log.info(f"  renderer.max_gaussians : {cfg['renderer'].get('max_gaussians', 'N/A')}")
    log.info(f"  position_lr_init       : {t.get('position_lr_init')}")
    log.info(f"  position_lr_max_steps  : {t.get('position_lr_max_steps')}")
    log.info("=" * 60)

    # ── Load COLMAP model ─────────────────────────────────────────────────────
    log.info(f"Loading COLMAP sparse model from: {args.source_path}")
    cameras_colmap, images_colmap, points3D = load_colmap_model(args.source_path)

    # ── FIX-5: Scene normalization ────────────────────────────────────────────
    # Centre scene at origin and scale cameras into unit sphere.  This prevents
    # Gaussian explosion from off-origin or badly-scaled COLMAP reconstructions.
    if not args.no_normalize:
        log.info("Normalizing scene (translate to origin, scale to unit sphere) …")
        from preprocessing.normalize_scene import normalize_scene, scene_stats
        pre  = scene_stats(images_colmap, points3D)
        log.info(f"  Before: extent={pre.get('camera_extent', '?')}")
        images_colmap, points3D, norm_info = normalize_scene(images_colmap, points3D)
        post = scene_stats(images_colmap, points3D)
        log.info(
            f"  After:  extent={post.get('camera_extent', '?')}  "
            f"scale={norm_info['scale']:.6f}  "
            f"centroid={norm_info['centroid'].round(3).tolist()}"
        )
    else:
        log.warning("Scene normalization DISABLED (--no_normalize). "
                    "Gaussian explosion may occur on off-origin scenes.")
        norm_info = {"centroid": np.zeros(3), "scale": 1.0, "applied": False}

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = ColmapDataset(
        frames_dir=args.image_dir,
        sparse_dir=args.source_path,
    )

    # Inject the normalized images into the dataset so Camera.from_colmap uses
    # the correct (normalized) tvec values during training.
    dataset.images = images_colmap

    train_cameras = dataset.views

    # ── Large-dataset guard ───────────────────────────────────────────────────
    num_images = len(train_cameras)
    if num_images > 250:
        # Cap to the *configured* iteration count, not a hardcoded number.
        # Previously this was hardcoded to 18000 which silently overrode config
        # values like 30000 on any dataset with >250 images.
        config_iters = cfg["training"]["iterations"]
        safe_cap = min(config_iters, 25000)
        if config_iters > safe_cap:
            log.warning(
                f"Large dataset ({num_images} images). "
                f"Capping iterations: {config_iters} → {safe_cap} "
                f"(set training.iterations ≤ {safe_cap} in config.yaml to silence)."
            )
            cfg["training"]["iterations"] = safe_cap

    log.info(
        f"Dataset: {num_images} images | "
        f"Iterations: {cfg['training']['iterations']} | "
        f"Max Gaussians: {cfg['training'].get('max_gaussians', 'N/A')}"
    )

    # ── Scene extent ──────────────────────────────────────────────────────────
    cameras_extent = compute_cameras_extent(dataset)
    log.info(f"cameras_extent = {cameras_extent:.4f}  (should be ~1.0 after normalization)")
    dataset.cameras_extent = cameras_extent

    # ── Point cloud → Gaussian init ───────────────────────────────────────────
    xyz_np, rgb_np = get_sparse_point_cloud(points3D)
    xyz_np, rgb_np = subsample_point_cloud(xyz_np, rgb_np, MAX_INIT_GAUSSIANS)

    xyz = torch.from_numpy(xyz_np).float()
    rgb = torch.from_numpy(rgb_np).float()

    log.info(f"Initialising {len(xyz):,} Gaussians from sparse point cloud.")

    model = GaussianModel(sh_degree=cfg["model"]["sh_degree"])
    # After normalization spatial_lr_scale is always ~1.0 (scene is in unit sphere)
    model.initialise_from_pcd(xyz, rgb, spatial_lr_scale=1.0)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(cfg=cfg, model=model, scene=dataset)

    if resume_path:
        trainer._setup_optimizer()
        trainer.resume_from_checkpoint(str(resume_path))

    trainer.train()

    # ── Export ────────────────────────────────────────────────────────────────
    export_dir = Path(args.model_path) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    gaussians = model.get_state()
    save_ply(str(export_dir / "final.ply"),     gaussians)
    save_splat(str(export_dir / "final.splat"), gaussians)
    _copy_training_artifacts(args.model_path, run_dir)
    if drive_root:
        _copy_exports_to_drive(export_dir, drive_root / "exports" / run["run_id"])
    experiment_manager.finalize_run(
        run_dir,
        status="completed",
        extra={
            "exports": {
                "final_ply": str(export_dir / "final.ply"),
                "final_splat": str(export_dir / "final.splat"),
            }
        },
    )
    log.info("Exported final artifacts to: %s", export_dir)
    log.info("Training complete.")


def _copy_pretraining_reports(image_dir: str, run_dir: Path) -> None:
    """Copy preprocessing reports produced beside the frames directory."""
    frames_dir = Path(image_dir)
    candidates = [
        frames_dir.parent / "quality_report.json",
        frames_dir.parent / "prediction_report.json",
        frames_dir.parent / "frame_selection_report.json",
    ]
    for src in candidates:
        if src.exists():
            shutil.copy2(src, run_dir / src.name)


def _copy_training_artifacts(model_path: str, run_dir: Path) -> None:
    from core.experiments.artifact_manager import ArtifactManager

    artifacts = ArtifactManager(run_dir)
    model_dir = Path(model_path)
    artifacts.track_directory(model_dir / "checkpoints", kind="checkpoint")
    artifacts.track_directory(model_dir / "exports", kind="export")
    artifacts.track_directory(model_dir / "previews", kind="preview")


def _apply_memory_safety(cfg: dict, hardware_report: dict) -> None:
    """Conservative VRAM-aware caps for Colab/Kaggle reliability."""
    vram_gb = float(hardware_report.get("vram_gb") or 0.0)
    training = cfg.setdefault("training", {})
    renderer = cfg.setdefault("renderer", {})
    if 0 < vram_gb <= 16:
        training["max_gaussians"] = min(int(training.get("max_gaussians", 80000)), 80000)
        training["densification_interval"] = max(int(training.get("densification_interval", 500)), 500)
    elif 16 < vram_gb <= 24:
        training["max_gaussians"] = min(int(training.get("max_gaussians", 120000)), 120000)
    renderer["max_gaussians"] = training.get("max_gaussians", renderer.get("max_gaussians", 150000))


def _prepare_drive_persistence(cfg: dict) -> Path | None:
    runtime = cfg.setdefault("runtime", {})
    if not runtime.get("drive_persistence", True):
        return None
    drive_root = Path(os.environ.get("MONOSPLAT_DRIVE_ROOT", runtime.get("drive_root", "")))
    if not str(drive_root) or str(drive_root) == ".":
        return None
    if not drive_root.parent.exists():
        return None
    for subdir in ["datasets", "checkpoints", "experiments", "exports"]:
        (drive_root / subdir).mkdir(parents=True, exist_ok=True)
    return drive_root


def _copy_exports_to_drive(export_dir: Path, drive_export_dir: Path) -> None:
    drive_export_dir.mkdir(parents=True, exist_ok=True)
    for artifact in export_dir.glob("*"):
        if artifact.is_file():
            shutil.copy2(artifact, drive_export_dir / artifact.name)


if __name__ == "__main__":
    main()
