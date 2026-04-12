"""
worker.py
Async pipeline worker for MonoSplat GOD MODE.

This function handles:
    1. Frame extraction (FFmpeg) - runs everywhere
    2. COLMAP pose estimation - runs everywhere
    3. Training - ONLY on GPU (Colab), otherwise prepares for Colab
    4. Export - only after training

The split workflow:
    Desktop (CPU) → Run steps 1-2, then STOP with "ready_for_colab" status
    Colab (GPU)   → Run step 3 (training), then step 4 (export)
"""

import json
import time
import traceback
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    ("EXTRACTING",  5,  "Extracting frames from video"),
    ("COLMAP",     35,  "Running COLMAP pose estimation"),
    ("TRAINING",   85,  "Training Gaussian Splat model (GPU required)"),
    ("EXPORTING",  95,  "Exporting .ply and .splat files"),
    ("READY",     100,  "Done"),
]


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _update_job(registry, job_id: str, **kwargs) -> None:
    try:
        registry.update_job(job_id, **kwargs)
    except Exception as e:
        print(f"[worker] registry update error: {e}")


def _set_stage(registry, job_id: str, stage: str, progress: int, message: str) -> None:
    print(f"[worker/{job_id}] [{stage}] {progress}% — {message}")
    _update_job(registry, job_id, status=stage.lower(), progress=progress, message=message)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _save_metrics(work_dir: Path, metrics: dict) -> None:
    path = work_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[worker] Metrics saved → {path}")


def _compute_psnr(pred, gt) -> float:
    import torch
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * (- __import__("math").log10(mse))


def _compute_ssim(pred, gt) -> float:
    try:
        from pytorch_msssim import ssim as pt_ssim
        return pt_ssim(
            pred.unsqueeze(0), gt.unsqueeze(0),
            data_range=1.0, size_average=True
        ).item()
    except ImportError:
        mu1 = pred.mean().item()
        mu2 = gt.mean().item()
        s1 = pred.std().item()
        s2 = gt.std().item()
        s12 = ((pred - mu1) * (gt - mu2)).mean().item()
        c1, c2 = 0.01**2, 0.03**2
        num = (2*mu1*mu2 + c1) * (2*s12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
        return num / (den + 1e-8)


# ---------------------------------------------------------------------------
# Main worker function
# ---------------------------------------------------------------------------

def run_pipeline(
    job_id:     str,
    input_path: str,
    work_dir:   str,
    config_path: str,
    registry,
    colmap_binary: str = "colmap",
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> None:
    """
    Full pipeline worker with GPU/CPU split awareness.

    On CPU (desktop): Runs extraction + COLMAP, then stops with "ready_for_colab" status.
    On GPU (Colab): Runs training + export to produce final .splat.
    """
    work_dir_path = Path(work_dir)
    input_path = Path(input_path)
    frames_dir = work_dir_path / "frames"
    colmap_dir = work_dir_path / "colmap"
    model_dir = work_dir_path / "models"

    for d in (frames_dir, colmap_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"job_id": job_id, "started_at": time.time()}

    # Detect if we're on a GPU (Colab) or CPU (desktop)
    from src.utils.env_detect import has_torch_gpu
    has_gpu = has_torch_gpu()
    print(f"[worker] GPU available: {has_gpu}")

    try:
        # ----------------------------------------------------------------
        # STAGE 1 — Frame extraction (always runs)
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXTRACTING", 5, "Extracting frames with FFmpeg…")

        from src.preprocessing.extract_frames import extract_from_video
        n_frames = extract_from_video(
            str(input_path),
            output_dir=str(frames_dir),
            fps=3,
            max_frames=400,
            blur_threshold=50.0,    # more lenient — keep more frames for COLMAP
            adaptive_sampling=False, # disable scene-change filter; use steady fps
        )
        metrics["num_frames"] = n_frames
        _update_job(registry, job_id, num_images=n_frames)
        _set_stage(registry, job_id, "EXTRACTING", 20, f"Extracted {n_frames} frames ✓")

        # ----------------------------------------------------------------
        # STAGE 2 — COLMAP (always runs)
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "COLMAP", 25, "Running COLMAP feature extraction…")
        t_colmap = time.time()

        def colmap_progress(step, line):
            _update_job(registry, job_id, message=f"[COLMAP] {step}: {line[:80]}")
            if on_progress:
                on_progress(step, line)

        from src.preprocessing.colmap_runner import run_colmap
        run_colmap(
            image_dir=str(frames_dir),
            output_dir=str(colmap_dir),
            colmap_binary=colmap_binary,
            on_progress=colmap_progress,
        )

        text_model = colmap_dir / "sparse_text"
        if not text_model.exists() or not any(text_model.iterdir()):
            raise RuntimeError(
                "COLMAP produced no model. "
                "Ensure 60%+ overlap and good lighting in your video."
            )

        metrics["colmap_time_s"] = round(time.time() - t_colmap, 1)
        _set_stage(registry, job_id, "COLMAP", 55, "COLMAP complete — poses estimated ✓")

        # ----------------------------------------------------------------
        # STAGE 3 — Training (GPU only, otherwise prepare for Colab)
        # ----------------------------------------------------------------
        if not has_gpu:
            # On desktop (CPU): stop here and wait for Colab training
            _set_stage(
                registry, job_id,
                "READY_FOR_COLAB", 60,
                f"✅ COLMAP complete! Ready for Colab training.\n"
                f"Zip work/{job_id}/frames and work/{job_id}/colmap and upload to Colab.\n"
                f"Then run the Colab notebook to train on GPU."
            )
            _update_job(
                registry, job_id,
                status="ready_for_colab",
                progress=60,
                message=f"COLMAP complete. Upload work/{job_id}/ to Colab for GPU training."
            )
            # Save paths for Colab pickup
            metrics["colmap_path"] = str(colmap_dir)
            metrics["frames_path"] = str(frames_dir)
            metrics["ready_for_colab"] = True
            _save_metrics(work_dir_path, metrics)
            return  # Stop here on desktop - no training on CPU

        # Continue with training on GPU (Colab)
        _set_stage(registry, job_id, "TRAINING", 60, "Starting Gaussian Splat training on GPU…")
        t_train = time.time()

        from src.utils.config_loader import load_config
        cfg = load_config(config_path)

        device = "cuda"
        print(f"[worker] Training device: {device}")

        # Load COLMAP model
        from src.preprocessing.utils import load_colmap_model
        cameras_colmap, images_colmap, points3d = load_colmap_model(str(text_model))

        import numpy as np
        import torch
        from PIL import Image as PILImage

        cfg_w = cfg.viewer.window_width
        cfg_h = cfg.viewer.window_height
        train_cameras, train_images = [], []

        for img_data in images_colmap.values():
            cam_data = cameras_colmap[img_data.camera_id]
            from src.renderer.camera import Camera as ViewerCamera
            cam = ViewerCamera.from_colmap(img_data, cam_data, cfg_w, cfg_h)
            img_file = frames_dir / img_data.name
            if not img_file.exists():
                img_file = frames_dir / Path(img_data.name).name
            if not img_file.exists():
                continue
            img = np.array(
                PILImage.open(img_file).convert("RGB").resize((cfg_w, cfg_h), PILImage.LANCZOS),
                dtype=np.float32,
            ) / 255.0
            train_cameras.append(cam)
            train_images.append(torch.from_numpy(img).permute(2, 0, 1))

        # Initialize model from COLMAP points
        from src.reconstruction.gaussian_model import GaussianModel
        from src.reconstruction.trainer import GaussianTrainer
        from src.renderer.renderer import GaussianRenderer

        xyzs = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
        rgbs = np.array([p.rgb for p in points3d.values()], dtype=np.float32) / 255.0

        model = GaussianModel(sh_degree=cfg.renderer.sh_degree)
        model.create_from_points(xyzs, rgbs)

        cfg.training.output_dir = str(model_dir / "gaussian")
        cfg.training.checkpoint_dir = str(model_dir / "checkpoints")

        renderer_obj = GaussianRenderer(
            width=cfg.viewer.window_width,
            height=cfg.viewer.window_height,
            bg_color=cfg.renderer.background_color,
            device=device,
        )

        trainer = GaussianTrainer(model, renderer_obj.render_torch, train_cameras, train_images, cfg)

        total_iters = cfg.training.iterations
        _last_pct = [55]

        def on_iter(it: int, loss: float) -> None:
            raw_pct = 60 + int((it / total_iters) * 25)
            if raw_pct > _last_pct[0]:
                _last_pct[0] = raw_pct
                _update_job(
                    registry, job_id,
                    progress=raw_pct,
                    message=f"Training… iter {it:,}/{total_iters:,}  loss={loss:.4f}",
                )

        trainer.train(on_iter_callback=on_iter)

        metrics["training_time_s"] = round(time.time() - t_train, 1)
        metrics["num_gaussians"] = len(model)
        _update_job(registry, job_id, num_gaussians=len(model))
        _set_stage(registry, job_id, "TRAINING", 85, f"Training complete — {len(model):,} Gaussians ✓")

        # ----------------------------------------------------------------
        # STAGE 4 — Export .ply + .splat
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXPORTING", 90, "Exporting .ply and .splat…")

        from src.utils.io_utils import save_ply, save_splat
        ply_path = model_dir / "gaussian" / f"{job_id}.ply"
        splat_path = model_dir / "gaussian" / f"{job_id}.splat"

        save_ply(str(ply_path), model.get_state())
        save_splat(str(splat_path), model.get_state())

        _update_job(
            registry, job_id,
            ply_path=str(ply_path),
            splat_path=str(splat_path),
        )
        _set_stage(registry, job_id, "EXPORTING", 95, "Files exported ✓")

        # ----------------------------------------------------------------
        # STAGE 5 — Evaluation metrics
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXPORTING", 97, "Computing quality metrics…")
        try:
            psnr_vals, ssim_vals = [], []
            sample = train_cameras[:min(5, len(train_cameras))]
            for i, cam in enumerate(sample):
                pred = renderer_obj.render_torch(model, cam).clamp(0, 1)
                gt = train_images[i].to(device).clamp(0, 1)
                psnr_vals.append(_compute_psnr(pred, gt))
                ssim_vals.append(_compute_ssim(pred, gt))

            metrics["psnr"] = round(sum(psnr_vals) / len(psnr_vals), 2)
            metrics["ssim"] = round(sum(ssim_vals) / len(ssim_vals), 4)
            print(f"[worker] PSNR={metrics['psnr']} dB  SSIM={metrics['ssim']}")
        except Exception as e:
            print(f"[worker] Metrics computation skipped: {e}")
            metrics["psnr"] = None
            metrics["ssim"] = None

        metrics["finished_at"] = time.time()
        metrics["total_time_s"] = round(metrics["finished_at"] - metrics["started_at"], 1)
        _save_metrics(work_dir_path, metrics)
        _update_job(registry, job_id, metrics=metrics)

        # ----------------------------------------------------------------
        # DONE
        # ----------------------------------------------------------------
        _set_stage(
            registry, job_id,
            "READY", 100,
            f"✅ Ready — {len(model):,} splats | "
            f"PSNR={metrics.get('psnr', 'n/a')} | "
            f"SSIM={metrics.get('ssim', 'n/a')}"
        )
        _update_job(registry, job_id, status="ready")

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[worker] PIPELINE FAILED for job {job_id}:\n{tb}")
        _update_job(
            registry, job_id,
            status="failed",
            progress=0,
            message="Pipeline failed — see server logs",
            error=str(exc),
        )
        metrics["error"] = str(exc)
        metrics["finished_at"] = time.time()
        _save_metrics(work_dir_path, metrics)
        raise