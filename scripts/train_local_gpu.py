"""
train_local_gpu.py
MonoSplat — Local GPU Training Script
======================================
Replaces the Colab notebook (monosplat_colab_gpu.ipynb) for running
training directly on your own machine with a CUDA-capable GPU.

Usage
-----
    # Typical run — point at an existing COLMAP job:
    python scripts/train_local_gpu.py --job_id <job_id>

    # With explicit paths:
    python scripts/train_local_gpu.py \\
        --colmap_dir  work/<job_id>/colmap/sparse_text \\
        --frames_dir  work/<job_id>/frames \\
        --output_dir  work/<job_id>/models/gaussian \\
        --job_id      <job_id>

    # Override GPU profile (auto-detected by default):
    python scripts/train_local_gpu.py --job_id <job_id> \\
        --iterations 10000 --max_gaussians 200000 --sh_degree 3 \\
        --width 800 --height 600

    # Resume from checkpoint:
    python scripts/train_local_gpu.py --job_id <job_id> \\
        --resume work/<job_id>/models/checkpoints/checkpoint_005000.pkl

    # Preview render after training (saved as preview_final.png):
    python scripts/train_local_gpu.py --job_id <job_id> --preview

Prerequisites
-------------
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install pillow pyyaml plyfile tqdm numpy

Optional (perceptual loss, ~10s extra):
    pip install lpips

Optional (CUDA rasterizer — 10-100x faster rendering):
    git clone --recurse-submodules https://github.com/graphdeco-inria/diff-gaussian-rasterization
    pip install ./diff-gaussian-rasterization

GPU Profile Auto-detection
--------------------------
    >= 20 GB VRAM  (RTX 3090/4090, A100) : 30k iters, 500k Gaussians, SH=3, 960x540
    >= 8 GB VRAM   (RTX 3070/3080/4070)  : 15k iters, 200k Gaussians, SH=3, 800x450
    >= 4 GB VRAM   (RTX 3060/2060)       : 7k  iters,  80k Gaussians, SH=1, 640x360
    < 4 GB / CPU   (fallback)            :  1k iters,  10k Gaussians, SH=0, 480x270
"""

from __future__ import annotations

import argparse
import functools
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve project root — script lives in scripts/, project root is one up
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# GPU profile table
# ---------------------------------------------------------------------------

def _pick_profile(vram_gb: float) -> dict:
    """Return training hyper-parameters based on available VRAM."""
    if vram_gb >= 20:
        return dict(
            iterations=30_000, max_gaussians=500_000, sh_degree=3,
            width=960, height=540, batch_size=5_000,
            densify_from=500,  densify_until=20_000, densify_interval=100,
            grad_threshold=0.0002, label="High-end (>=20 GB)",
        )
    elif vram_gb >= 8:
        return dict(
            iterations=15_000, max_gaussians=100_000, sh_degree=3,
            width=960, height=540, batch_size=3_000,
            densify_from=500,  densify_until=12_000, densify_interval=100,
            grad_threshold=0.0002, label="Mid-range (8-20 GB)",
        )
    elif vram_gb >= 4:
        return dict(
            iterations=15_000, max_gaussians=100_000, sh_degree=3,
            width=960, height=540, batch_size=1_500,
            densify_from=500,  densify_until=12_000, densify_interval=100,
            grad_threshold=0.0002, label="Entry-level (4-8 GB)",
        )
    else:
        return dict(
            iterations=1_000, max_gaussians=10_000, sh_degree=0,
            width=480, height=270, batch_size=500,
            densify_from=200,  densify_until=800,   densify_interval=50,
            grad_threshold=0.0005, label="CPU / low-VRAM fallback",
        )


# ---------------------------------------------------------------------------
# Step 1 — Environment check
# ---------------------------------------------------------------------------

def check_environment() -> str:
    """Verify GPU, return device string."""
    print("\n" + "=" * 62)
    print("  Step 1 — Environment Check")
    print("=" * 62)

    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split("\n")[:13]:
            print(line)
    else:
        print("  nvidia-smi not found — will try PyTorch CUDA anyway")

    assert sys.version_info >= (3, 9), f"Python 3.9+ required, got {sys.version}"
    print(f"\n  Python {sys.version.split()[0]}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    if not torch.cuda.is_available():
        print("\n  CUDA not available — running on CPU (very slow)")
        return "cpu"

    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"\n  GPU  : {props.name}")
    print(f"   VRAM : {vram_gb:.1f} GB")
    print(f"   CUDA : {torch.version.cuda}")

    torch.cuda.empty_cache()
    return "cuda"


# ---------------------------------------------------------------------------
# Step 2 — Install / check CUDA rasterizer (optional)
# ---------------------------------------------------------------------------

def check_cuda_rasterizer() -> bool:
    """Return True if diff-gaussian-rasterization is available."""
    print("\n" + "=" * 62)
    print("  Step 2 — CUDA Rasterizer Check")
    print("=" * 62)
    try:
        import diff_gaussian_rasterization as dgr  # noqa: F401
        print("  diff-gaussian-rasterization available -> fast render path")
        return True
    except ImportError:
        print("  diff-gaussian-rasterization NOT installed")
        print("   -> Software renderer will be used (~100x slower)")
        print("   To install:")
        print("     git clone --recurse-submodules \\")
        print("       https://github.com/graphdeco-inria/diff-gaussian-rasterization")
        print("     pip install ./diff-gaussian-rasterization")
        return False


# ---------------------------------------------------------------------------
# Step 3 — Verify COLMAP output
# ---------------------------------------------------------------------------

def verify_colmap(colmap_dir: str, frames_dir: str) -> None:
    """Check that COLMAP files exist and have content."""
    import glob

    print("\n" + "=" * 62)
    print("  Step 3 — Verify COLMAP Output")
    print("=" * 62)
    print(f"  COLMAP dir : {colmap_dir}")
    print(f"  Frames dir : {frames_dir}")

    all_ok = True
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        path = os.path.join(colmap_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data_lines = [l for l in f if not l.startswith("#") and l.strip()]
            status = "  " if data_lines else "  EMPTY"
            print(f"  {status}  {fname:<20} {len(data_lines):>6} data lines")
            if not data_lines:
                all_ok = False
        else:
            print(f"  X  {fname:<20} MISSING")
            all_ok = False

    frame_files = (
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpg")) +
        glob.glob(os.path.join(frames_dir, "*.jpeg"))
    )
    print(f"\n  Frames found : {len(frame_files)}")

    if not all_ok:
        raise RuntimeError(
            "COLMAP output is incomplete.\n"
            "Run COLMAP on your desktop first:\n"
            "  uvicorn src.pipeline.server:app --reload --port 8000\n"
            "  Then upload your video and wait for COLMAP to finish."
        )
    if len(frame_files) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")
    if len(frame_files) < 20:
        warnings.warn(
            f"Only {len(frame_files)} frames — reconstruction may be sparse. "
            "Consider re-running with more frames."
        )
    print(f"\n  COLMAP verified. {len(frame_files)} frames ready.")


# ---------------------------------------------------------------------------
# Step 4 — Apply model patches (same as Colab notebook)
# ---------------------------------------------------------------------------

def apply_patches() -> None:
    """Patch GaussianModel for robust kNN and no-grad density ops."""
    from src.reconstruction import gaussian_model as _gm
    from src.reconstruction.gaussian_model import GaussianModel

    # ── Batched GPU kNN ────────────────────────────────────────────────
    # The naive implementation computes a full N×N distance matrix which
    # requires N² * 4 bytes of RAM — for 80k points that is ~25 GB and
    # will crash on any consumer machine.  This version processes the
    # matrix in row-chunks of BATCH so peak memory is only BATCH×N*4
    # (~640 MB at BATCH=2048, N=80k), well within 4 GB VRAM.
    def _knn_robust(pts: torch.Tensor, k: int = 3) -> torch.Tensor:
        N = pts.shape[0]
        if N == 0:
            return torch.tensor([], device=pts.device)
        k2 = min(k, N - 1)
        if k2 == 0:
            return torch.zeros(N, device=pts.device)

        # Run on GPU when available — cdist is significantly faster there
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pts = pts.to(compute_device)

        BATCH   = 2048  # rows per chunk — safe for 4 GB VRAM
        results = torch.zeros(N, device=compute_device)

        for start in range(0, N, BATCH):
            end   = min(start + BATCH, N)
            chunk = pts[start:end]           # (B, 3)
            D     = torch.cdist(chunk, pts)  # (B, N) — only B rows at once

            # Zero out self-distances (the diagonal block for this chunk)
            self_cols = torch.arange(start, end, device=compute_device)
            D[torch.arange(end - start, device=compute_device), self_cols] = float("inf")

            top = torch.topk(D, k=k2, largest=False, dim=1).values  # (B, k)
            results[start:end] = top.mean(dim=1).clamp(min=1e-7)

        return results.cpu()

    if hasattr(_gm, "_knn_mean_dist"):
        _gm._knn_mean_dist = _knn_robust

    # ── Wrap density-control ops under no_grad ─────────────────────────
    # Prevents autograd graph pollution from in-place Gaussian ops.
    def _no_grad(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            with torch.no_grad():
                return fn(*a, **kw)
        return wrap

    for method_name in (
        "reset_opacity", "densify_and_prune", "prune_points",
        "densify_and_clone", "densify_and_split",
    ):
        if hasattr(GaussianModel, method_name):
            setattr(GaussianModel, method_name, _no_grad(getattr(GaussianModel, method_name)))


# ---------------------------------------------------------------------------
# Step 5 — Load data
# ---------------------------------------------------------------------------

def load_data(colmap_dir: str, frames_dir: str, cfg, W: int, H: int):
    """Load COLMAP model and training images."""
    from PIL import Image
    from src.preprocessing.utils import load_colmap_model
    from src.renderer.camera import Camera as ViewerCamera

    print(f"\n  Loading COLMAP from : {colmap_dir}")
    cameras_colmap, images_colmap, points3d = load_colmap_model(colmap_dir)
    print(f"  {len(images_colmap)} registered images, {len(points3d):,} 3D points")

    print(f"  Loading images at {W}x{H}...")
    train_cameras, train_images, missing = [], [], []

    for img_data in sorted(images_colmap.values(), key=lambda i: i.name):
        cam_data = cameras_colmap[img_data.camera_id]
        cam      = ViewerCamera.from_colmap(img_data, cam_data, W, H)

        img_path = Path(frames_dir) / img_data.name
        if not img_path.exists():
            img_path = Path(frames_dir) / Path(img_data.name).name
        if not img_path.exists():
            missing.append(img_data.name)
            continue

        img = Image.open(img_path).convert("RGB")
        if W and H:
            img = img.resize((W, H), Image.LANCZOS)
        img_np = np.array(img, dtype=np.float32) / 255.0
        train_cameras.append(cam)
        train_images.append(torch.from_numpy(img_np).permute(2, 0, 1).cpu())

    if missing:
        print(f"  {len(missing)} frames not found (first 3: {missing[:3]})")
    if not train_cameras:
        raise RuntimeError(f"No images loaded. Check frames_dir={frames_dir}")

    xyzs = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
    rgbs = np.array([p.rgb for p in points3d.values()], dtype=np.float32) / 255.0

    print(f"  {len(train_cameras)} cameras, {len(xyzs):,} initial points")
    return train_cameras, train_images, xyzs, rgbs


# ---------------------------------------------------------------------------
# Step 6 — Train
# ---------------------------------------------------------------------------

def train(args, cfg, profile: dict, device: str) -> tuple:
    """
    Main training loop — adapted from Cell 8 of the Colab notebook.
    Returns (model, renderer, train_cameras, train_images).
    """
    from src.reconstruction.gaussian_model import GaussianModel
    from src.reconstruction.trainer import GaussianTrainer
    from src.reconstruction.loss import combined_loss
    from src.renderer.renderer import GaussianRenderer

    W   = args.width   or profile["width"]
    H   = args.height  or profile["height"]
    SH  = args.sh_degree if args.sh_degree is not None else profile["sh_degree"]
    BS  = profile["batch_size"]
    ITERS      = args.iterations    or profile["iterations"]
    MAX_GAUSS  = args.max_gaussians or profile["max_gaussians"]
    D_FROM     = profile["densify_from"]
    D_UNTIL    = profile["densify_until"]
    D_INTERVAL = profile["densify_interval"]
    GRAD_THR   = profile["grad_threshold"]

    print("\n" + "=" * 62)
    print("  Step 5 — Training")
    print("=" * 62)
    print(f"  GPU profile   : {profile['label']}")
    print(f"  iterations    : {ITERS:,}")
    print(f"  max_gaussians : {MAX_GAUSS:,}")
    print(f"  sh_degree     : {SH}")
    print(f"  resolution    : {W}x{H}")
    print(f"  batch_size    : {BS}")
    print(f"  densify       : iter {D_FROM} -> {D_UNTIL}, every {D_INTERVAL}")
    print(f"  grad_threshold: {GRAD_THR}")
    print(f"  output_dir    : {args.output_dir}")

    # Apply config overrides
    cfg.training.iterations              = ITERS
    cfg.renderer.max_gaussians           = MAX_GAUSS
    cfg.renderer.sh_degree               = SH
    cfg.training.output_dir              = args.output_dir
    cfg.training.checkpoint_dir          = args.ckpt_dir
    cfg.training.densify_from_iter       = D_FROM
    cfg.training.densify_until_iter      = D_UNTIL
    cfg.training.densification_interval  = D_INTERVAL
    cfg.training.densify_grad_threshold  = GRAD_THR
    if not hasattr(cfg, "viewer"):
        cfg.viewer = type("Config", (), {})()
    cfg.viewer.window_width  = W
    cfg.viewer.window_height = H
    # Write training resolution separately — viewer dims must not drive data loading
    if not hasattr(cfg, "training"):
        cfg.training = type("Config", (), {})()

    save_every    = getattr(cfg.training, "save_every", 1000)
    lambda_dssim  = getattr(cfg.training, "lambda_dssim", 0.2)
    percent_dense = getattr(cfg.training, "percent_dense", 0.01)

    # Load data
    train_cameras, train_images, xyzs, rgbs = load_data(
        args.colmap_dir, args.frames_dir, cfg, W, H
    )

    # Cap initial cloud at 125,000 points — preserves geometry vs the old 50k cap.
    # The raw COLMAP cloud may have 125k+ points; discarding them starves the
    # Gaussian model of geometry and limits densification headroom.
    MAX_INIT_POINTS = 125000
    if len(xyzs) > MAX_INIT_POINTS:
        idx  = np.random.choice(len(xyzs), MAX_INIT_POINTS, replace=False)
        xyzs = xyzs[idx];  rgbs = rgbs[idx]
        print(f"  Raw cloud: {len(xyzs) + (len(xyzs) - MAX_INIT_POINTS):,} points")
        print(f"  ⚠️  Downsampled to {MAX_INIT_POINTS:,} points")

    # Init model
    model = GaussianModel(sh_degree=SH)
    model = model.to(device)
    model.create_from_points(xyzs, rgbs)

    # Renderer
    if device == "cuda":
        torch.cuda.empty_cache();  gc.collect()
        torch.backends.cudnn.benchmark = True

    renderer = GaussianRenderer(
        width=W, height=H,
        bg_color=cfg.renderer.background_color,
        device=device,
        batch_size=BS,
        use_gsplat=getattr(cfg.renderer, "use_gsplat", getattr(cfg.renderer, "use_cuda_rasterizer", True)),
    )

    # Trainer
    trainer = GaussianTrainer(
        model=model,
        renderer=renderer.render_torch,
        train_cameras=train_cameras,
        train_images=train_images,
        cfg=cfg,
    )

    # Resume from checkpoint
    start_iter = 0
    if args.resume:
        start_iter = trainer.resume_from_checkpoint(args.resume)
        print(f"  Resumed from iteration {start_iter}")

    # VRAM estimate
    if device == "cuda":
        bytes_per_gauss = (3 + 1 + 3 + 3 + 4 + 48) * 4
        vram_est_mb     = len(model) * bytes_per_gauss / 1e6
        free_vram_gb    = torch.cuda.mem_get_info()[0] / 1e9
        total_vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  VRAM estimate : ~{vram_est_mb:.0f} MB for {len(model):,} Gaussians")
        print(f"  Free VRAM now : {free_vram_gb:.1f} GB / {total_vram_gb:.1f} GB total")
        if vram_est_mb > free_vram_gb * 1000 * 0.8:
            print("  Close to VRAM limit — lower --max_gaussians if you get OOM")

    # Training loop
    from tqdm import tqdm

    optimizer     = trainer.optimizer
    n             = len(model)
    grad_accum    = torch.zeros(n, device=device)
    grad_denom    = torch.zeros(n, device=device)
    nan_count     = 0
    oom_count     = 0
    loss_val      = 0.0

    print(f"\n  Starting training ({ITERS:,} iterations)...\n")
    pbar = tqdm(range(start_iter, ITERS), desc="Training", dynamic_ncols=True)

    for it in pbar:
        idx    = np.random.randint(len(train_cameras))
        camera = train_cameras[idx]
        gt_img = train_images[idx].to(device, non_blocking=True)

        try:
            optimizer.zero_grad(set_to_none=True)
            rendered = renderer.render_torch(model, camera)
            loss     = combined_loss(rendered, gt_img, lambda_ssim=lambda_dssim)

            if torch.isnan(loss):
                nan_count += 1
                if nan_count <= 5 or nan_count % 50 == 0:
                    tqdm.write(f"[{it:5d}]   NaN loss (total {nan_count}), skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if model._positions.grad is not None:
                with torch.no_grad():
                    gn = model._positions.grad.norm(dim=1)
                    if gn.shape[0] == grad_accum.shape[0]:
                        grad_accum += gn
                        grad_denom += 1.0

            optimizer.step()
            trainer._apply_position_lr_decay(it, ITERS)
            loss_val = loss.item()

        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache();  gc.collect()
            tqdm.write(f"[{it:5d}] OOM #{oom_count} — emergency prune to 60% cap")
            emergency = int(MAX_GAUSS * 0.6)
            cur = len(model)
            if cur > emergency:
                n_prune = cur - emergency
                with torch.no_grad():
                    _, low = model.opacities.squeeze().topk(n_prune, largest=False)
                    mask   = torch.zeros(cur, dtype=torch.bool, device=device)
                    mask[low] = True
                model.prune_points(mask)
                trainer._setup_optimizer()
                optimizer  = trainer.optimizer
                n          = len(model)
                grad_accum = torch.zeros(n, device=device)
                grad_denom = torch.zeros(n, device=device)
            continue

        # SH degree scheduling
        if it % 1000 == 0:
            model.oneup_sh_degree()

        # Densification
        if D_FROM <= it <= D_UNTIL and it % D_INTERVAL == 0:
            trainer._grad_accum = grad_accum.clone()
            trainer._grad_denom = grad_denom.clone()
            trainer._densify_and_prune(grad_threshold=GRAD_THR, percent_dense=percent_dense)
            optimizer  = trainer.optimizer
            n          = len(model)
            grad_accum = torch.zeros(n, device=device)
            grad_denom = torch.zeros(n, device=device)
            tqdm.write(f"[{it:5d}] Densified -> {n:,} Gaussians")

        # Checkpoint
        if it > 0 and it % save_every == 0:
            trainer._save(it)

        # VRAM periodic clear
        if device == "cuda" and it % 500 == 0:
            torch.cuda.empty_cache()

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "N":    f"{len(model):,}",
            "NaN":  nan_count,
            "OOM":  oom_count,
        })

    trainer._save(ITERS)

    print(f"\n  Training complete.")
    print(f"   Gaussians   : {len(model):,}")
    print(f"   Final loss  : {loss_val:.4f}")
    print(f"   NaN skipped : {nan_count}")
    print(f"   OOM events  : {oom_count}")
    print(f"   Output dir  : {args.output_dir}")

    return model, renderer, train_cameras, train_images


# ---------------------------------------------------------------------------
# Step 7 — Export .splat
# ---------------------------------------------------------------------------

def export_splat(output_dir: str, job_id: str) -> None:
    """Convert the latest .ply checkpoint to browser-ready .splat format."""
    import glob
    from src.utils.io_utils import save_splat, load_ply

    print("\n" + "=" * 62)
    print("  Step 6 — Export .splat")
    print("=" * 62)

    ply_files = sorted(glob.glob(os.path.join(output_dir, "*.ply")), key=os.path.getmtime)
    if not ply_files:
        print(f"  No .ply files in {output_dir}")
        return

    latest_ply = ply_files[-1]
    print(f"  Latest PLY : {latest_ply}  ({os.path.getsize(latest_ply)/1e6:.1f} MB)")

    try:
        gaussians  = load_ply(latest_ply)
        splat_path = os.path.join(output_dir, f"{job_id}.splat")
        save_splat(splat_path, gaussians)
        print(f"  Exported : {splat_path}  ({os.path.getsize(splat_path)/1e6:.1f} MB)")
    except Exception as e:
        print(f"  .splat export failed: {e}")
        print("     You can still use the .ply — drag it to https://supersplat.playcanvas.com")

    import shutil
    ply_dest = os.path.join(output_dir, f"{job_id}.ply")
    if os.path.abspath(latest_ply) != os.path.abspath(ply_dest):
        shutil.copy2(latest_ply, ply_dest)
    print(f"  PLY ready : {ply_dest}")


# ---------------------------------------------------------------------------
# Step 8 — Optional preview render
# ---------------------------------------------------------------------------

def render_preview(model, renderer, train_cameras, train_images, output_dir: str, job_id: str) -> None:
    """Render a preview frame and save as PNG (requires matplotlib)."""
    print("\n" + "=" * 62)
    print("  Step 7 — Preview Render")
    print("=" * 62)

    try:
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
    except ImportError:
        print("  matplotlib not installed — skipping preview (pip install matplotlib)")
        return

    idx    = len(train_cameras) // 2
    camera = train_cameras[idx]

    with torch.no_grad():
        rendered = renderer.render_torch(model, camera)

    img_np = rendered.detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    gt_np  = (train_images[idx].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(gt_np);   axes[0].set_title("Ground Truth");   axes[0].axis("off")
    axes[1].imshow(img_np);  axes[1].set_title("Rendered Splat"); axes[1].axis("off")
    plt.suptitle(f"MonoSplat Preview — Job {job_id} — Camera {idx}", fontsize=12)
    plt.tight_layout()

    preview_path = os.path.join(output_dir, "preview_final.png")
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.show()
    PILImage.fromarray(img_np).save(preview_path)
    print(f"  Preview saved : {preview_path}")
    print(f"     Gaussians  : {len(model):,}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MonoSplat — Local GPU Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--job_id",     default=None,
                   help="Job ID — paths are inferred from work/<job_id>/ if set")
    p.add_argument("--colmap_dir", default=None,
                   help="COLMAP sparse_text directory")
    p.add_argument("--frames_dir", default=None,
                   help="Extracted frames directory")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for .ply / .splat / checkpoints")
    p.add_argument("--config",     default="config/config.yaml",
                   help="Config YAML path (default: config/config.yaml)")
    p.add_argument("--resume",     default=None,
                   help="Checkpoint .pkl path to resume from")

    p.add_argument("--iterations",    type=int, default=None,
                   help="Override iteration count (default: auto by VRAM)")
    p.add_argument("--max_gaussians", type=int, default=None,
                   help="Override max Gaussians (default: auto by VRAM)")
    p.add_argument("--sh_degree",     type=int, default=None, choices=[0,1,2,3],
                   help="SH degree 0-3 (default: auto by VRAM)")
    p.add_argument("--width",  type=int, default=None, help="Render width")
    p.add_argument("--height", type=int, default=None, help="Render height")

    p.add_argument("--preview", action="store_true",
                   help="Render a preview frame after training")
    p.add_argument("--skip_verify", action="store_true",
                   help="Skip COLMAP verification (useful if you know files are good)")

    args = p.parse_args()

    if args.job_id:
        work = PROJECT_ROOT / "work" / args.job_id
        args.colmap_dir = args.colmap_dir or str(work / "colmap" / "sparse_text")
        args.frames_dir = args.frames_dir or str(work / "frames")
        args.output_dir = args.output_dir or str(work / "models" / "gaussian")
    else:
        missing = [f for f, v in [
            ("--colmap_dir", args.colmap_dir),
            ("--frames_dir", args.frames_dir),
            ("--output_dir", args.output_dir),
        ] if not v]
        if missing:
            p.error(f"Provide --job_id OR explicit paths: {', '.join(missing)}")

    args.ckpt_dir = str(Path(args.output_dir) / "checkpoints")

    return args


def main() -> None:
    args = parse_args()

    config_path = args.config if os.path.isabs(args.config) else str(PROJECT_ROOT / args.config)

    print("\n" + "=" * 62)
    print("  MonoSplat — Local GPU Training")
    print("=" * 62)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Job ID       : {args.job_id or '(explicit paths)'}")
    print(f"  COLMAP dir   : {args.colmap_dir}")
    print(f"  Frames dir   : {args.frames_dir}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Config       : {config_path}")

    device = check_environment()
    check_cuda_rasterizer()

    if not args.skip_verify:
        verify_colmap(args.colmap_dir, args.frames_dir)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda" else 0.0
    profile = _pick_profile(vram_gb)
    print(f"\n  Auto-selected profile : {profile['label']}")

    apply_patches()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    from src.utils.config_loader import load_config
    cfg = load_config(config_path)

    model, renderer, train_cameras, train_images = train(args, cfg, profile, device)

    export_splat(args.output_dir, args.job_id or "model")

    if args.preview:
        render_preview(model, renderer, train_cameras, train_images,
                       args.output_dir, args.job_id or "model")

    job_id = args.job_id or "model"
    print("\n" + "=" * 62)
    print("  Done! Next steps:")
    print("=" * 62)
    print(f"  1. Your files are in:  {args.output_dir}")
    print(f"     {job_id}.splat  <- drag to https://supersplat.playcanvas.com")
    print(f"     {job_id}.ply    <- or use locally")
    if args.job_id:
        print(f"\n  2. Update models/registry.json for job {job_id}:")
        print(f'       "status":     "ready",')
        print(f'       "splat_path": "work/{job_id}/models/gaussian/{job_id}.splat",')
        print(f'       "ply_path":   "work/{job_id}/models/gaussian/{job_id}.ply"')
        print(f"\n  3. View in browser:")
        print(f"       http://localhost:8000/viewer/{job_id}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()