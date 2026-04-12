"""
train.py
Standalone training script — Object / Product / Architecture mode.

CLI arguments aligned with LeoDarcy/360GS train.py
---------------------------------------------------
    -s / --source_path   Path to COLMAP data (replaces --colmap_dir)
    -m / --model_path    Where to save output (replaces --output_dir)
    --eval               Hold out test cameras for evaluation (360GS flag)
    --iterations         Override iteration count
    --sh_degree          Spherical harmonics degree (0–3; default 3)
    --resume             Checkpoint path to resume from

Usage examples (matching 360GS style)
--------------------------------------
    # Full training on a captured object:
    python train.py -s data/colmap_output -m models/shoe_01 --eval

    # Quick CPU test:
    python train.py -s data/colmap_output -m models/test --iterations 500

    # Resume from checkpoint:
    python train.py -s data/colmap_output -m models/shoe_01 --resume models/shoe_01/checkpoints/checkpoint_015000.pkl
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config
from src.preprocessing.utils import load_colmap_model
from src.reconstruction.gaussian_model import GaussianModel
from src.reconstruction.trainer import GaussianTrainer
from src.renderer.camera import Camera as ViewerCamera
from src.renderer.renderer import GaussianRenderer
from src.utils.io_utils import save_splat, load_ply


# ---------------------------------------------------------------------------
# Data loading with eval split (matches 360GS --eval flag behaviour)
# ---------------------------------------------------------------------------

def load_training_data(colmap_dir: str, image_dir: str, cfg, eval_split: bool = False):
    """
    Load COLMAP model and corresponding images.
    With eval_split=True, holds out every 8th image as a test camera
    (same strategy as 360GS / 3DGS reference code).

    Returns:
        train_cameras, train_images, test_cameras, test_images,
        point_cloud (N,3), point_colors (N,3)
    """
    cameras_colmap, images_colmap, points3d = load_colmap_model(colmap_dir)

    W = cfg.viewer.window_width
    H = cfg.viewer.window_height

    all_cameras = []
    all_images  = []
    image_dir   = Path(image_dir)
    skipped     = 0

    print("[train] Loading images…")
    img_list = sorted(images_colmap.values(), key=lambda i: i.name)

    for img_data in tqdm(img_list, desc="Images"):
        cam_data = cameras_colmap[img_data.camera_id]
        cam      = ViewerCamera.from_colmap(img_data, cam_data, W, H)

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
        all_cameras.append(cam)
        all_images.append(torch.from_numpy(img).permute(2, 0, 1).cpu())

    if skipped:
        print(f"[train] ⚠  {skipped} images not found — skipped.")

    # ── Eval split (360GS / 3DGS convention: every 8th image is test) ──
    if eval_split and len(all_cameras) >= 16:
        test_idx   = list(range(0, len(all_cameras), 8))
        train_idx  = [i for i in range(len(all_cameras)) if i not in set(test_idx)]
        test_cameras  = [all_cameras[i] for i in test_idx]
        test_images   = [all_images[i]  for i in test_idx]
        train_cameras = [all_cameras[i] for i in train_idx]
        train_images  = [all_images[i]  for i in train_idx]
        print(f"[train] Eval split: {len(train_cameras)} train, {len(test_cameras)} test")
    else:
        train_cameras = all_cameras
        train_images  = all_images
        test_cameras  = []
        test_images   = []
        if eval_split:
            print("[train] ⚠  Too few images for eval split — using all for training.")

    xyzs = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
    rgbs = np.array([p.rgb for p in points3d.values()], dtype=np.float32) / 255.0

    print(f"[train] {len(train_cameras)} train cameras | {len(xyzs):,} initial points")
    return train_cameras, train_images, test_cameras, test_images, xyzs, rgbs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MonoSplat Object Training — 360GS-aligned CLI"
    )

    # ── 360GS-style flags ─────────────────────────────────────────────
    parser.add_argument("-s", "--source_path", required=True,
                        help="COLMAP sparse_text directory (data/colmap_output/sparse_text)")
    parser.add_argument("-m", "--model_path",  default="models/gaussian",
                        help="Output directory for PLY / .splat / checkpoints")
    parser.add_argument("--eval", action="store_true",
                        help="Hold out every 8th image as test set (360GS convention)")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical harmonics degree 0–3 (default: 3)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override total training iterations")

    # ── MonoSplat flags (kept for backward compat) ────────────────────
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--image_dir", default=None,
                        help="Training image directory (default: source_path/../processed)")
    parser.add_argument("--resume",    default=None)

    args = parser.parse_args()
    cfg  = load_config(args.config)

    # Resolve image directory
    if args.image_dir:
        image_dir = args.image_dir
    else:
        # Infer from source_path: colmap_output/sparse_text → ../processed
        image_dir = str(Path(args.source_path).parent.parent / "processed")

    # Apply CLI overrides
    cfg.training.output_dir     = args.model_path
    cfg.training.checkpoint_dir = str(Path(args.model_path) / "checkpoints")
    cfg.renderer.sh_degree      = args.sh_degree

    if args.iterations is not None:
        cfg.training.iterations = args.iterations

    # Safety caps for free-tier Colab T4
    cfg.training.iterations    = min(cfg.training.iterations,    30000)
    cfg.renderer.max_gaussians = min(cfg.renderer.max_gaussians, 1000000)

    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 62)
    print("  MonoSplat — Object / Product / Architecture Mode")
    print("=" * 62)
    print(f"  Source (COLMAP): {args.source_path}")
    print(f"  Image dir      : {image_dir}")
    print(f"  Output         : {cfg.training.output_dir}")
    print(f"  Device         : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else 'CPU'}")
    print(f"  SH degree      : {args.sh_degree}")
    print(f"  Iterations     : {cfg.training.iterations:,}")
    print(f"  Eval split     : {'yes (every 8th image)' if args.eval else 'no'}")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────────
    train_cams, train_imgs, test_cams, test_imgs, init_xyz, init_rgb = \
        load_training_data(args.source_path, image_dir, cfg, eval_split=args.eval)

    if not train_cams:
        print("[train] ✗  No valid training cameras. Check --source_path and image_dir.")
        sys.exit(1)

    # ── Initialise model ──────────────────────────────────────────────
    model = GaussianModel(sh_degree=args.sh_degree)
    model.create_from_points(init_xyz, init_rgb)

    # ── Renderer ──────────────────────────────────────────────────────
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache() if device == "cuda" else None

    renderer_obj = GaussianRenderer(
        width=cfg.viewer.window_width,
        height=cfg.viewer.window_height,
        bg_color=cfg.renderer.background_color,
        device=device,
        batch_size=getattr(cfg.renderer, "batch_size", 5000),
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = GaussianTrainer(
        model=model,
        renderer=renderer_obj.render_torch,
        train_cameras=train_cams,
        train_images=train_imgs,
        cfg=cfg,
        test_cameras=test_cams,
        test_images=test_imgs,
    )

    start_iter = 0
    if args.resume:
        start_iter = trainer.resume_from_checkpoint(args.resume)

    trainer.train(start_iter=start_iter)

    # ── Export .splat ──────────────────────────────────────────────────
    output_dir = Path(cfg.training.output_dir)
    ply_files  = sorted(output_dir.glob("*.ply"))
    if ply_files:
        latest_ply = ply_files[-1]
        splat_path = latest_ply.with_suffix(".splat")
        gaussians  = load_ply(str(latest_ply))
        save_splat(str(splat_path), gaussians)
        print(f"[train] Exported .splat → {splat_path}")

    # ── Print eval summary (matches 360GS metrics output) ─────────────
    if trainer.eval_log:
        print("\n[Eval Summary]")
        for rec in trainer.eval_log[-5:]:
            print(f"  iter={rec['iter']:6d}  PSNR={rec['psnr']} dB  SSIM={rec['ssim']}")

    print(f"\n[train] ✅  Done!  Outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()