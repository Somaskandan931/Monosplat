#!/usr/bin/env python3
"""
debug_reconstruction.py
=======================
Standalone end-to-end pipeline for debugging MonoSplat reconstruction.

PURPOSE
-------
When the server pipeline breaks, you have NO way to reproduce it without
starting FastAPI, uploading via the browser, and waiting for the queue.
This script removes all of that: it runs the full pipeline sequentially,
prints every metric to stdout, and saves metrics.json.

WORKFLOW
--------
1. Extract frames from video
2. Filter (blur / duplicate rejection)
3. Run COLMAP (with hard validation gates)
4. Validate sparse cloud
5. Initialize Gaussians from sparse points
6. Run short training (configurable iterations)
7. Export preview renders
8. Save metrics.json and .ply splat

NO DEPENDENCIES ON:
  - FastAPI
  - queues
  - WebSockets
  - frontend state
  - cloud storage
  - async workers

USAGE
-----
    python scripts/debug_reconstruction.py video.mp4 --output debug_out/run01

    # Short smoke-test run (500 iters)
    python scripts/debug_reconstruction.py video.mp4 --iters 500

    # Skip re-extraction if frames already exist
    python scripts/debug_reconstruction.py video.mp4 --skip-extraction --output debug_out/run01

    # Dump all intermediate metrics without training
    python scripts/debug_reconstruction.py video.mp4 --no-train

This script is the GROUND TRUTH for whether the pipeline works.
Until this produces a reliable splat on 3 fixed test videos,
do not touch the server.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# ── Ensure repo root is on the path so `src.*` imports work ────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MonoSplat standalone debug reconstruction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", help="Input video file (.mp4 / .mov / …)")
    p.add_argument(
        "--output", "-o",
        default="debug_out/run",
        help="Output directory (default: debug_out/run)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Training iterations (default: 1000 for fast smoke-test; use 15000 for full)",
    )
    p.add_argument(
        "--colmap-binary",
        default="colmap",
        help="Path to COLMAP binary (default: colmap)",
    )
    p.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="COLMAP reconstruction quality (default: medium)",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip frame extraction if frames/ already exists in --output",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Stop after COLMAP — print reconstruction metrics only",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Training device: cuda | cpu (default: cuda)",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Optional job ID for metrics.json (default: derived from video filename)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_extract(video_path: Path, frames_dir: Path) -> dict:
    """Stage 1+2: Extract and filter frames from video."""
    print("\n" + "="*60)
    print("STAGE 1/5: Frame extraction")
    print("="*60)

    from src.preprocessing.extract_frames import extract_from_video

    frames_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # extract_from_video returns the number of frames kept after filtering
    kept = extract_from_video(str(video_path), str(frames_dir))

    elapsed = time.time() - t0
    all_frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    print(f"\n[debug] Extraction: {kept} frames kept in {elapsed:.1f}s → {frames_dir}")

    return {
        "frame_count": len(all_frames),
        "filtered_frames": kept,
        "elapsed_s": round(elapsed, 2),
    }


def stage_colmap(frames_dir: Path, colmap_dir: Path, binary: str, quality: str) -> dict:
    """Stage 3: Run COLMAP with hard validation gates."""
    print("\n" + "="*60)
    print("STAGE 2/5: COLMAP sparse reconstruction")
    print("="*60)

    from src.preprocessing.colmap_runner import run_colmap

    t0 = time.time()
    # run_colmap raises RuntimeError on any hard failure — do NOT catch it here.
    # The whole point of the debug script is to surface failures loudly.
    run_colmap(
        image_dir=str(frames_dir),
        output_dir=str(colmap_dir),
        colmap_binary=binary,
        quality=quality,
    )
    elapsed = time.time() - t0

    # Read metrics from the output files
    text_dir = colmap_dir / "sparse_text"
    points3d_txt = text_dir / "points3D.txt"
    images_txt = text_dir / "images.txt"

    n_points = 0
    if points3d_txt.exists():
        n_points = sum(
            1 for line in open(points3d_txt)
            if line.strip() and not line.startswith("#")
        )

    registered = 0
    all_images = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    total = len(all_images)
    if images_txt.exists():
        idx = 0
        with open(images_txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if idx % 2 == 0:
                    registered += 1
                idx += 1

    print(f"\n[debug] COLMAP: {registered}/{total} images, {n_points:,} 3D points, {elapsed:.1f}s")

    return {
        "registered_images": registered,
        "total_images": total,
        "registration_ratio": round(registered / max(total, 1), 3),
        "sparse_points": n_points,
        "elapsed_s": round(elapsed, 2),
    }


def stage_init_gaussians(colmap_dir: Path, device: str) -> dict:
    """Stage 4: Initialise Gaussian model from sparse point cloud."""
    print("\n" + "="*60)
    print("STAGE 3/5: Gaussian initialisation")
    print("="*60)

    from src.reconstruction.gaussian_model import GaussianModel
    from src.utils.io_utils import load_point_cloud

    text_dir = colmap_dir / "sparse_text"
    points3d_txt = text_dir / "points3D.txt"

    if not points3d_txt.exists():
        raise FileNotFoundError(f"points3D.txt not found at {points3d_txt}")

    t0 = time.time()
    pts, colors = load_point_cloud(str(points3d_txt))
    print(f"[debug] Loaded {len(pts):,} points from sparse cloud")

    model = GaussianModel(sh_degree=3)
    model.initialise_from_point_cloud(pts, colors, device=device)
    elapsed = time.time() - t0

    n_gaussians = len(model)
    print(f"[debug] Initialised {n_gaussians:,} Gaussians in {elapsed:.1f}s on {device}")

    return {
        "initial_gaussians": n_gaussians,
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "model": model,  # passed to training stage
    }


def stage_train(
    model,
    colmap_dir: Path,
    output_dir: Path,
    iters: int,
    device: str,
) -> dict:
    """Stage 5: Short training run."""
    print("\n" + "="*60)
    print(f"STAGE 4/5: Training ({iters} iterations)")
    print("="*60)

    try:
        import yaml
        from types import SimpleNamespace

        cfg_path = REPO_ROOT / "config" / "config.yaml"
        with open(cfg_path) as f:
            raw_cfg = yaml.safe_load(f)

        # Build a minimal config namespace for the trainer
        def _ns(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
            return d

        cfg = _ns(raw_cfg)
        cfg.training.iterations = iters
        cfg.training.output_dir = str(output_dir / "gaussian")
        cfg.training.checkpoint_dir = str(output_dir / "checkpoints")

    except Exception as e:
        print(f"[debug] Could not load config.yaml: {e} — using defaults")
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                iterations=iters,
                output_dir=str(output_dir / "gaussian"),
                checkpoint_dir=str(output_dir / "checkpoints"),
                learning_rate=SimpleNamespace(
                    position=0.00016, feature=0.0025, opacity=0.05,
                    scaling=0.005, rotation=0.001, position_final=0.0000016,
                ),
                densify_from_iter=500,
                densify_until_iter=iters,
                densification_interval=100,
                opacity_reset_interval=3000,
                percent_dense=0.01,
                densify_grad_threshold=0.0002,
                lambda_dssim=0.2,
                save_every=max(500, iters // 5),
                eval_every=max(200, iters // 10),
            ),
            renderer=SimpleNamespace(
                background_color=[1.0, 1.0, 1.0],
                sh_degree=3,
                max_gaussians=1_000_000,
                batch_size=5000,
            ),
        )

    from src.utils.io_utils import load_cameras_and_images
    from src.renderer.renderer import GaussianRenderer
    from src.reconstruction.trainer import GaussianTrainer

    text_dir = colmap_dir / "sparse_text"
    frames_dir = colmap_dir.parent / "frames"

    print("[debug] Loading cameras from COLMAP output…")
    train_cameras, train_images = load_cameras_and_images(
        str(text_dir), str(frames_dir)
    )
    print(f"[debug] Loaded {len(train_cameras)} training cameras")

    renderer = GaussianRenderer(cfg)
    trainer = GaussianTrainer(
        model=model,
        renderer=renderer,
        train_cameras=train_cameras,
        train_images=train_images,
        cfg=cfg,
    )

    t0 = time.time()
    loss_curve = []

    def _on_iter(it, loss, n_gauss):
        loss_curve.append(float(loss))
        if it % max(1, iters // 10) == 0:
            print(f"  iter {it:5d}/{iters}  loss={loss:.4f}  gaussians={n_gauss:,}")

    trainer.train(on_iter_callback=_on_iter)
    elapsed = time.time() - t0

    # Export .ply
    ply_path = output_dir / "splat.ply"
    from src.utils.io_utils import save_ply
    save_ply(model, str(ply_path))
    print(f"\n[debug] Splat saved → {ply_path}")

    final_gaussians = len(model)
    nan_count = getattr(trainer, "nan_count", 0)
    final_loss = loss_curve[-1] if loss_curve else None

    print(f"[debug] Training complete: {final_gaussians:,} Gaussians, loss={final_loss:.4f}, {elapsed:.1f}s")

    return {
        "final_gaussians": final_gaussians,
        "nan_iterations": nan_count,
        "loss_curve": loss_curve[::max(1, len(loss_curve)//50)],  # downsample for JSON
        "final_loss": round(final_loss, 6) if final_loss else None,
        "elapsed_s": round(elapsed, 2),
        "ply_path": str(ply_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()

    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    job_id = args.job_id or video_path.stem
    output_dir = Path(args.output).resolve()
    frames_dir = output_dir / "frames"
    colmap_dir = output_dir / "colmap"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# MonoSplat Debug Pipeline")
    print(f"# video : {video_path}")
    print(f"# output: {output_dir}")
    print(f"# iters : {args.iters}")
    print(f"# device: {args.device}")
    print(f"{'#'*60}\n")

    # Import after path setup
    from src.utils.metrics import PipelineMetrics

    metrics = PipelineMetrics(job_id=job_id)
    wall_start = time.time()

    try:
        # ---- Stage 1: Frame extraction -----------------------------------
        if args.skip_extraction and frames_dir.exists():
            all_frames = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
            print(f"[debug] Skipping extraction — {len(all_frames)} frames already in {frames_dir}")
            frame_stats = {"frame_count": len(all_frames), "filtered_frames": len(all_frames)}
        else:
            frame_stats = stage_extract(video_path, frames_dir)

        metrics.set_frame_metrics(
            frame_count=frame_stats.get("frame_count", 0),
            filtered_frames=frame_stats.get("filtered_frames", 0),
        )

        # ---- Stage 2: COLMAP ---------------------------------------------
        colmap_stats = stage_colmap(frames_dir, colmap_dir, args.colmap_binary, args.quality)
        metrics.set_reconstruction_metrics(
            registered_images=colmap_stats["registered_images"],
            total_images=colmap_stats["total_images"],
            sparse_points=colmap_stats["sparse_points"],
        )

        if args.no_train:
            print("\n[debug] --no-train flag set. Stopping after reconstruction.")
            metrics.mark_success(wall_seconds=time.time() - wall_start)
            metrics.save(work_dir=str(output_dir))
            print(metrics.summary())
            return

        # ---- Stage 3: Gaussian init --------------------------------------
        init_stats = stage_init_gaussians(colmap_dir, args.device)
        model = init_stats.pop("model")

        # ---- Stage 4: Training -------------------------------------------
        train_stats = stage_train(model, colmap_dir, output_dir, args.iters, args.device)
        metrics.set_training_metrics(
            initial_gaussians=init_stats["initial_gaussians"],
            final_gaussians=train_stats["final_gaussians"],
            nan_iterations=train_stats["nan_iterations"],
            loss_curve=train_stats["loss_curve"],
        )

        # ---- Done --------------------------------------------------------
        total_elapsed = time.time() - wall_start
        metrics.mark_success(wall_seconds=round(total_elapsed, 2))

    except Exception as exc:
        total_elapsed = time.time() - wall_start
        metrics.mark_failed(str(exc), wall_seconds=round(total_elapsed, 2))
        metrics.save(work_dir=str(output_dir))
        print(f"\n{'!'*60}", file=sys.stderr)
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        print(f"{'!'*60}", file=sys.stderr)
        print(metrics.summary())
        sys.exit(1)

    metrics.save(work_dir=str(output_dir))

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(metrics.summary())
    print(f"\nOutputs:")
    print(f"  metrics : {output_dir}/metrics.json")
    if not args.no_train:
        print(f"  splat   : {output_dir}/splat.ply")
    print(f"  frames  : {frames_dir}")
    print(f"  colmap  : {colmap_dir}")


if __name__ == "__main__":
    main()
