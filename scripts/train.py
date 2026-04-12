"""
train.py
Standalone training script: loads COLMAP data + images, trains GaussianModel,
saves PLY and .splat outputs.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow running as `python -m scripts.train` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config
from src.preprocessing.utils import load_colmap_model
from src.reconstruction.gaussian_model import GaussianModel
from src.reconstruction.trainer import GaussianTrainer
from src.renderer.camera import Camera as ViewerCamera
from src.renderer.renderer import GaussianRenderer
from src.utils.io_utils import save_splat, load_ply


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(colmap_dir: str, image_dir: str, cfg):
    """
    Load COLMAP model and corresponding ground-truth images.

    Returns:
        train_cameras : list of ViewerCamera
        train_images  : list of (3, H, W) float32 torch.Tensor  in [0, 1]
        point_cloud   : (N, 3) float32 initial positions
        point_colors  : (N, 3) float32 RGB in [0, 1]
    """
    cameras_colmap, images_colmap, points3d = load_colmap_model(colmap_dir)

    W = cfg.viewer.window_width
    H = cfg.viewer.window_height

    train_cameras: list = []
    train_images:  list = []
    image_dir = Path(image_dir)
    skipped   = 0

    print("[train] Loading training images…")
    for img_data in tqdm(list(images_colmap.values()), desc="Images"):
        cam_data = cameras_colmap[img_data.camera_id]

        # Build camera from COLMAP extrinsics / intrinsics
        cam = ViewerCamera.from_colmap(img_data, cam_data, W, H)

        # Resolve image path (try with and without sub-directory prefix)
        img_path = image_dir / img_data.name
        if not img_path.exists():
            img_path = image_dir / Path(img_data.name).name
        if not img_path.exists():
            skipped += 1
            continue

        img = np.array(
            Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        train_cameras.append(cam)
        train_images.append(torch.from_numpy(img).permute(2, 0, 1).cpu())   # (3, H, W) — stays on CPU to save VRAM

    if skipped:
        print(f"[train] ⚠  {skipped} images not found — skipped.")

    # Initial point cloud from COLMAP sparse reconstruction
    xyzs = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
    rgbs = np.array([p.rgb for p in points3d.values()], dtype=np.float32) / 255.0

    print(
        f"[train] {len(train_cameras)} cameras  |  "
        f"{len(xyzs):,} initial points"
    )
    return train_cameras, train_images, xyzs, rgbs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a 3D Gaussian Splatting model.")
    parser.add_argument("--config",     default="config/config.yaml",
                        help="Config file path (default: config/config.yaml)")
    parser.add_argument("--colmap_dir", required=True,
                        help="COLMAP sparse_text directory (data/colmap_output/sparse_text)")
    parser.add_argument("--image_dir",  required=True,
                        help="Training image directory (data/processed)")
    parser.add_argument("--output_dir", default="models/gaussian",
                        help="PLY output directory (default: models/gaussian)")
    parser.add_argument("--resume",     default=None,
                        help="Checkpoint path to resume from (models/checkpoints/*.pkl)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override paths from CLI args
    cfg.training.output_dir     = args.output_dir
    cfg.training.checkpoint_dir = str(Path(args.output_dir).parent / "checkpoints")

    # Memory-safe caps — respect config values but never exceed free-tier T4 limits
    cfg.training.iterations    = min(cfg.training.iterations,    30000)
    cfg.renderer.max_gaussians = min(cfg.renderer.max_gaussians, 200000)
    cfg.renderer.sh_degree     = min(cfg.renderer.sh_degree,     1)

    # Ensure output directories exist
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Gaussian XR — Training")
    print("=" * 62)
    print(f"  COLMAP dir   : {args.colmap_dir}")
    print(f"  Image dir    : {args.image_dir}")
    print(f"  PLY output   : {cfg.training.output_dir}")
    print(f"  Checkpoints  : {cfg.training.checkpoint_dir}")
    print(f"  Device       : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Iterations   : {cfg.training.iterations:,}")
    print(f"  Max Gaussians: {cfg.renderer.max_gaussians:,}")
    print("=" * 62)

    # Load data
    train_cameras, train_images, init_xyz, init_rgb = load_training_data(
        args.colmap_dir, args.image_dir, cfg
    )
    if not train_cameras:
        print("[train] ✗  No valid training cameras found. Check --image_dir and --colmap_dir.")
        sys.exit(1)

    # Initialise model
    model = GaussianModel(sh_degree=cfg.renderer.sh_degree)
    model.create_from_points(init_xyz, init_rgb)

    # Renderer (used inside trainer for loss computation)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer_obj = GaussianRenderer(
        width=cfg.viewer.window_width,
        height=cfg.viewer.window_height,
        bg_color=cfg.renderer.background_color,
        device=device,
        batch_size=getattr(cfg.renderer, "batch_size", 5000),
    )

    # Trainer
    trainer = GaussianTrainer(
        model=model,
        renderer=renderer_obj.render_torch,
        train_cameras=train_cameras,
        train_images=train_images,
        cfg=cfg,
    )

    # Resume from checkpoint
    start_iter = 0
    if args.resume:
        start_iter = trainer.resume_from_checkpoint(args.resume)

    # Train
    trainer.train(start_iter=start_iter)

    # Export final .splat file alongside PLY files
    output_dir = Path(cfg.training.output_dir)
    ply_files  = sorted(output_dir.glob("*.ply"))
    if ply_files:
        latest_ply   = ply_files[-1]
        splat_path   = latest_ply.with_suffix(".splat")
        from src.utils.io_utils import load_ply
        gaussians = load_ply(str(latest_ply))
        save_splat(str(splat_path), gaussians)
        print(f"[train] Exported .splat → {splat_path}")

    print(f"\n[train] ✅  Done!  Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()