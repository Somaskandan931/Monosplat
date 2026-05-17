"""
worker.py
Async pipeline worker for MonoSplat.

Pipeline stages
---------------
1. EXTRACTING — FFmpeg frame extraction with blur/motion filtering
2. COLMAP     — SfM pose estimation (exhaustive matcher, OPENCV model)
3. TRAINING   — Gaussian Splat training via gsplat CUDA backend
                 (or paused at ready_for_colab if no GPU is available)
4. EXPORTING  — Export .ply (Gaussian model) + .splat (browser viewer)
5. READY      — Done

Split workflow
--------------
  Desktop (CPU) → stages 1–2, stop at "ready_for_colab"
  Colab (GPU)   → stage 3 training, stage 4 export
"""

import json
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from src.utils.console import configure_console_encoding
configure_console_encoding()


STAGES = [
    ("EXTRACTING",       5,  "Extracting frames from video"),
    ("COLMAP",          35,  "Running COLMAP pose estimation"),
    ("TRAINING",        85,  "Training Gaussian Splat model (GPU required)"),
    ("EXPORTING",       95,  "Exporting .ply and .splat files"),
    ("READY",          100,  "Done"),
]


def _update_job(registry, job_id: str, **kwargs) -> None:
    try:
        registry.update_job(job_id, **kwargs)
    except Exception as e:
        print(f"[worker] registry update error: {e}")


def _set_stage(registry, job_id: str, stage: str, progress: int, message: str) -> None:
    print(f"[worker/{job_id}] [{stage}] {progress}% — {message}")
    _update_job(registry, job_id, status=stage.lower(), progress=progress, message=message)


def _save_metrics(work_dir: Path, metrics: dict) -> None:
    import numpy as np
    import torch

    def _convert(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, torch.Tensor):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path = work_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_convert)
    print(f"[worker] Metrics saved → {path}")


def run_cmd_silent(cmd: list, step_name: str) -> int:
    import subprocess
    print(f"\n[worker] ▶  {step_name}")
    print("  " + " ".join(str(c) for c in cmd))
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"[worker] {step_name}: {line}")
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"[worker] {step_name} failed: {e}")
        return 1


def _compute_psnr(pred, gt) -> float:
    from src.reconstruction.loss import psnr_metric
    return psnr_metric(pred, gt).item()


def _compute_ssim(pred, gt) -> float:
    try:
        from pytorch_msssim import ssim as pt_ssim
        return pt_ssim(
            pred.unsqueeze(0), gt.unsqueeze(0),
            data_range=1.0, size_average=True
        ).item()
    except ImportError:
        mu1 = pred.mean().item(); mu2 = gt.mean().item()
        s1  = pred.std().item();  s2  = gt.std().item()
        s12 = ((pred - mu1) * (gt - mu2)).mean().item()
        c1, c2 = 0.01**2, 0.03**2
        num = (2*mu1*mu2 + c1) * (2*s12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
        return num / (den + 1e-8)


def _count_registered(text_model: Path) -> tuple:
    """Count registered images from COLMAP images.txt."""
    images_txt = text_model / "images.txt"
    if not images_txt.exists():
        return 0, 0
    registered = 0
    data_line_idx = 0
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if data_line_idx % 2 == 0:
                registered += 1
            data_line_idx += 1
    total = data_line_idx // 2
    return registered, total


def run_pipeline(
    job_id:      str,
    input_path:  str,
    work_dir:    str,
    config_path: str,
    registry,
    colmap_binary: str = "colmap",
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> None:
    work_dir_path = Path(work_dir)
    input_path    = Path(input_path)
    frames_dir    = work_dir_path / "frames"
    colmap_dir    = work_dir_path / "colmap"
    model_dir     = work_dir_path / "models"

    for d in (frames_dir, colmap_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"job_id": job_id, "started_at": time.time()}

    from src.utils.config_loader import load_config
    from src.utils.env_detect import has_torch_gpu, should_run_dense_mvs

    cfg = load_config(config_path)
    has_gpu = has_torch_gpu()
    print(f"[worker] GPU available: {has_gpu}")

    video_fps            = getattr(cfg.data, "video_fps", None)
    max_frames           = getattr(cfg.data, "max_frames", 600)
    colmap_binary        = getattr(cfg.colmap, "binary_path", colmap_binary)
    colmap_quality       = getattr(cfg.colmap, "quality", "medium")
    colmap_camera_model  = getattr(cfg.colmap, "camera_model", "OPENCV")
    colmap_single_camera = getattr(cfg.colmap, "single_camera", True)

    _VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    try:
        # ----------------------------------------------------------------
        # STAGE 1 — Frame extraction
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXTRACTING", 5, "Extracting frames…")

        from src.preprocessing.extract_frames import (
            extract_from_video, copy_images, estimate_motion,
            validate_image_resolution,
        )

        if input_path.is_dir():
            n_frames = copy_images(str(input_path), str(frames_dir))
        elif input_path.suffix.lower() in _VIDEO_EXT:
            n_frames = extract_from_video(
                str(input_path),
                output_dir=str(frames_dir),
                fps=float(video_fps) if video_fps is not None else None,
                max_frames=int(max_frames),
                blur_threshold=80.0,
                adaptive_sampling=False,
            )
        else:
            raise RuntimeError(f"Unsupported input: {input_path.suffix}. Upload .mp4/.mov/.avi or image folder.")

        _update_job(registry, job_id, num_images=n_frames)
        metrics["num_frames"] = n_frames

        if n_frames < 10:
            raise RuntimeError(
                f"Only {n_frames} usable frames extracted. "
                "Upload a longer video (>5 seconds) with slow, steady movement around the object."
            )

        # Validate resolution before COLMAP
        validate_image_resolution(str(frames_dir))

        # Motion check
        motion_score = estimate_motion(str(frames_dir))
        metrics["motion_score"] = round(motion_score, 3)
        warnings = []
        
        if motion_score < 1.0:
            warnings.append({
                "type": "motion",
                "severity": "error",
                "message": f"Motion score too low ({motion_score:.2f}). Walk around the object slowly — do not keep the camera still."
            })
            raise RuntimeError(
                f"Motion score too low ({motion_score:.2f}). "
                "Walk around the object slowly — do not keep the camera still."
            )
        elif motion_score > 80.0:
            warnings.append({
                "type": "motion",
                "severity": "warning",
                "message": f"High motion ({motion_score:.1f}) — results may be blurry. Shoot slower next time."
            })
            print(f"[worker] ⚠  High motion ({motion_score:.1f}) — results may be blurry. Shoot slower next time.")

        # Exposure validation (advisory — never blocks the pipeline)
        try:
            from src.preprocessing.extract_frames import validate_exposure
            exp = validate_exposure(str(frames_dir))
            metrics["exposure_ok"]        = exp["ok"]
            metrics["exposure_overexp"]   = exp["overexposed"]
            metrics["exposure_underexp"]  = exp["underexposed"]
            if exp["warning"]:
                warnings.append({
                    "type": "exposure",
                    "severity": "warning",
                    "message": exp["warning"]
                })
                _update_job(registry, job_id,
                    message=f"⚠ Exposure issue detected: {exp['overexposed']} overexposed, "
                            f"{exp['underexposed']} underexposed frames. Continuing…")
        except Exception as _exp_err:
            print(f"[worker] Exposure check skipped: {_exp_err}")
        
        # Save warnings to job metadata
        if warnings:
            _update_job(registry, job_id, warnings=warnings)

        _set_stage(registry, job_id, "EXTRACTING", 20,
                   f"{n_frames} frames extracted (motion score: {motion_score:.1f}) ✓")

        # ----------------------------------------------------------------
        # STAGE 2 — COLMAP SfM
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "COLMAP", 25, "Running COLMAP feature extraction…")
        t_colmap = time.time()

        from src.preprocessing.colmap_runner import run_colmap

        def colmap_progress(step: str, line: str) -> None:
            _update_job(registry, job_id, message=f"COLMAP [{step}]: {line[:80]}")
            if on_progress:
                on_progress(step, line)

        run_colmap(
            image_dir=str(frames_dir),
            output_dir=str(colmap_dir),
            colmap_binary=colmap_binary,
            camera_model=colmap_camera_model,
            single_camera=colmap_single_camera,
            use_gpu=True,
            quality=colmap_quality,
            on_progress=colmap_progress,
        )

        text_model = colmap_dir / "sparse_text"

        # Auto-retry if registration is poor
        registered, _ = _count_registered(text_model)
        registration_ratio = registered / max(n_frames, 1)
        metrics["registered_frames"]   = registered
        metrics["registration_ratio"]  = round(registration_ratio, 3)

        if registration_ratio < 0.5 and text_model.exists():
            print(f"[worker] [AUTO-FIX] {registration_ratio*100:.0f}% registered — retrying with low quality…")
            _set_stage(registry, job_id, "COLMAP", 40,
                       f"Auto-retry: {registration_ratio*100:.0f}% registered — relaxing thresholds…")
            import shutil
            shutil.rmtree(colmap_dir, ignore_errors=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            run_colmap(
                image_dir=str(frames_dir),
                output_dir=str(colmap_dir),
                colmap_binary=colmap_binary,
                camera_model=colmap_camera_model,
                single_camera=colmap_single_camera,
                use_gpu=True,
                quality="low",
                on_progress=colmap_progress,
            )
            registered, _ = _count_registered(text_model)
            registration_ratio = registered / max(n_frames, 1)
            metrics["registered_frames_retry"] = registered
            print(f"[worker] After retry: {registered}/{n_frames} ({registration_ratio*100:.0f}%)")

        if not text_model.exists() or not any(text_model.iterdir()):
            raise RuntimeError(
                "COLMAP produced no model. "
                "Ensure 60%+ overlap and good lighting. "
                "Walk around the object slowly with it centred in frame."
            )

        metrics["colmap_time_s"] = round(time.time() - t_colmap, 1)
        _set_stage(registry, job_id, "COLMAP", 55,
                   f"COLMAP complete — {registered}/{n_frames} frames registered ✓")

        # Optional dense reconstruction (MVS)
        if registered >= 10 and should_run_dense_mvs():
            _set_stage(registry, job_id, "COLMAP", 57, "Running dense reconstruction…")
            dense_dir  = colmap_dir / "dense"
            sparse_0   = colmap_dir / "sparse" / "0"
            dense_dir.mkdir(parents=True, exist_ok=True)
            try:
                rc_u = run_cmd_silent([
                    colmap_binary, "image_undistorter",
                    "--image_path",  str(frames_dir),
                    "--input_path",  str(sparse_0),
                    "--output_path", str(dense_dir),
                    "--output_type", "COLMAP",
                    "--max_image_size", "1000",
                ], "Dense: image_undistorter")
                if rc_u == 0:
                    rc_p = run_cmd_silent([
                        colmap_binary, "patch_match_stereo",
                        "--workspace_path",   str(dense_dir),
                        "--workspace_format", "COLMAP",
                        "--PatchMatchStereo.geom_consistency", "1",
                    ], "Dense: patch_match_stereo")
                    if rc_p == 0:
                        dense_ply = dense_dir / "fused.ply"
                        rc_f = run_cmd_silent([
                            colmap_binary, "stereo_fusion",
                            "--workspace_path",   str(dense_dir),
                            "--workspace_format", "COLMAP",
                            "--input_type",       "geometric",
                            "--output_path",      str(dense_ply),
                        ], "Dense: stereo_fusion")
                        if rc_f == 0 and dense_ply.exists():
                            metrics["dense_ply"] = str(dense_ply)
                            print(f"[worker] ✅ Dense point cloud: {dense_ply.stat().st_size/1e6:.1f} MB")
            except Exception as e:
                print(f"[worker] Dense reconstruction skipped: {e}")

        # ----------------------------------------------------------------
        # STAGE 3 — Training (GPU only)
        # ----------------------------------------------------------------
        if not has_gpu:
            _update_job(registry, job_id, status="ready_for_colab", progress=60,
                        message=f"COLMAP complete. Upload work/{job_id}/ to Colab for GPU training.")
            metrics["ready_for_colab"] = True
            metrics["colmap_path"]     = str(colmap_dir)
            metrics["frames_path"]     = str(frames_dir)
            _save_metrics(work_dir_path, metrics)
            return

        _set_stage(registry, job_id, "TRAINING", 60, "Starting Gaussian Splat training on GPU…")
        t_train = time.time()

        import numpy as np
        import torch
        from PIL import Image as PILImage

        from src.preprocessing.utils import load_colmap_model
        from src.renderer.camera import Camera as ViewerCamera
        from src.reconstruction.gaussian_model import GaussianModel
        from src.reconstruction.trainer import GaussianTrainer
        from src.renderer.renderer import GaussianRenderer

        device = "cuda"
        torch.backends.cudnn.benchmark = True

        cameras_colmap, images_colmap, points3d = load_colmap_model(str(text_model))

        cfg_w = cfg.viewer.window_width
        cfg_h = cfg.viewer.window_height
        train_cameras, train_images = [], []

        for img_data in images_colmap.values():
            cam_data = cameras_colmap[img_data.camera_id]
            cam      = ViewerCamera.from_colmap(img_data, cam_data, cfg_w, cfg_h)
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
            train_images.append(torch.from_numpy(img).permute(2, 0, 1).cpu())

        xyzs = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
        rgbs = np.array([p.rgb for p in points3d.values()], dtype=np.float32) / 255.0

        # Boost iterations for sparse point clouds
        num_points = len(xyzs)
        if num_points < 10_000:
            cfg.training.iterations = max(cfg.training.iterations, 60_000)
            print(f"[worker] Sparse cloud ({num_points} pts) — boosting iters to {cfg.training.iterations}")

        cfg.training.output_dir     = str(model_dir / "gaussian")
        cfg.training.checkpoint_dir = str(model_dir / "checkpoints")

        model = GaussianModel(sh_degree=cfg.renderer.sh_degree)
        model = model.to(device)
        model.create_from_points(xyzs, rgbs)

        renderer_obj = GaussianRenderer(
            width=cfg.viewer.window_width,
            height=cfg.viewer.window_height,
            bg_color=cfg.renderer.background_color,
            device=device,
            use_gsplat=True,   # use gsplat when available
        )

        trainer = GaussianTrainer(
            model, renderer_obj, train_cameras, train_images, cfg
        )

        total_iters  = cfg.training.iterations
        _last_pct    = [55]
        preview_dir  = work_dir_path / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Pick a fixed camera for consistent previews (first train cam)
        _preview_cam = train_cameras[0] if train_cameras else None
        _preview_every = max(1000, total_iters // 10)   # ~10 previews per run

        def on_iter(it: int, loss: float) -> None:
            raw_pct = 60 + int((it / total_iters) * 25)
            if raw_pct > _last_pct[0]:
                _last_pct[0] = raw_pct
                _update_job(
                    registry, job_id,
                    progress=raw_pct,
                    message=f"Training… iter {it:,}/{total_iters:,}  loss={loss:.4f}",
                )

            # Render preview frame at key milestones
            if _preview_cam is not None and it > 0 and it % _preview_every == 0:
                try:
                    from PIL import Image as _PIL_Image
                    prev_np = renderer_obj.render(model, _preview_cam)
                    prev_path = preview_dir / f"iter_{it:06d}.jpg"
                    _PIL_Image.fromarray(prev_np).save(str(prev_path), quality=82)
                    # Keep only the latest symlink for fast API serving
                    latest = preview_dir / "latest.jpg"
                    if latest.exists() or latest.is_symlink():
                        latest.unlink()
                    import shutil
                    shutil.copy2(str(prev_path), str(latest))
                    print(f"[worker] Preview saved → {prev_path.name}")
                except Exception as _pe:
                    pass   # previews are best-effort; never crash training

        trainer.train(on_iter_callback=on_iter)

        metrics["training_time_s"] = round(time.time() - t_train, 1)
        metrics["num_gaussians"]   = len(model)
        _update_job(registry, job_id, num_gaussians=len(model))
        _set_stage(registry, job_id, "TRAINING", 85, f"Training complete — {len(model):,} Gaussians ✓")

        # ----------------------------------------------------------------
        # STAGE 4 — Export
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXPORTING", 90, "Exporting .ply and .splat…")

        from src.utils.io_utils import save_ply, save_splat, save_spz, save_splat_chunks
        gaussian_dir = model_dir / "gaussian"
        gaussian_dir.mkdir(parents=True, exist_ok=True)

        ply_path   = gaussian_dir / f"{job_id}.ply"
        splat_path = gaussian_dir / f"{job_id}.splat"
        spz_path   = gaussian_dir / f"{job_id}.spz"
        chunks_dir = gaussian_dir / f"{job_id}_chunks"

        state = model.get_state()

        _set_stage(registry, job_id, "EXPORTING", 91, "Saving .ply…")
        save_ply(str(ply_path), state)

        _set_stage(registry, job_id, "EXPORTING", 93, "Saving .splat…")
        save_splat(str(splat_path), state)

        # Compressed .spz (optional — best-effort, non-blocking)
        try:
            _set_stage(registry, job_id, "EXPORTING", 94, "Compressing .spz…")
            save_spz(str(spz_path), state)
        except Exception as e:
            print(f"[worker] .spz export skipped: {e}")
            spz_path = None

        # Chunked streaming export (enables progressive loading in viewers)
        try:
            n_splats = len(model)
            if n_splats > 10_000:
                _set_stage(registry, job_id, "EXPORTING", 95, "Building streaming chunks…")
                save_splat_chunks(str(chunks_dir), state, chunk_size=50_000)
        except Exception as e:
            print(f"[worker] Chunk export skipped: {e}")

        _update_job(registry, job_id, ply_path=str(ply_path), splat_path=str(splat_path))
        _set_stage(registry, job_id, "EXPORTING", 96, "Files exported ✓")

        # ── Cloud storage upload (Stage 5) ─────────────────────────────────────
        try:
            cloud_cfg = getattr(cfg, "cloud_storage", None)
            if cloud_cfg and getattr(cloud_cfg, "enabled", False):
                from src.utils.cloud_storage import get_cloud_storage, upload_job_to_cloud
                
                storage_type = getattr(cloud_cfg, "type", "local")
                storage_config = {
                    "type": storage_type,
                    "bucket": getattr(getattr(cloud_cfg, storage_type, {}), "bucket", "monosplat-jobs"),
                }
                
                if storage_type == "s3":
                    s3_cfg = getattr(cloud_cfg, "s3", {})
                    storage_config["region"] = getattr(s3_cfg, "region", "us-east-1")
                    storage_config["aws_access_key_id"] = getattr(s3_cfg, "aws_access_key_id", None)
                    storage_config["aws_secret_access_key"] = getattr(s3_cfg, "aws_secret_access_key", None)
                elif storage_type == "gcs":
                    gcs_cfg = getattr(cloud_cfg, "gcs", {})
                    storage_config["credentials_path"] = getattr(gcs_cfg, "credentials_path", None)
                elif storage_type == "local":
                    local_cfg = getattr(cloud_cfg, "local", {})
                    storage_config["local_path"] = getattr(local_cfg, "path", "cloud_storage")
                
                storage = get_cloud_storage(storage_config)
                cloud_urls = upload_job_to_cloud(job_id, str(work_dir_path), storage)
                
                # Update job with cloud URLs
                _update_job(registry, job_id, cloud_urls=cloud_urls)
                print(f"[worker] Cloud upload complete: {len(cloud_urls)} files")
        except Exception as cloud_err:
            print(f"[worker] Cloud upload skipped: {cloud_err}")

        # ── AI Layer analysis (Stage 7) ─────────────────────────────────────────────
        try:
            ai_cfg = getattr(cfg, "ai_layer", None)
            if ai_cfg and getattr(ai_cfg, "enabled", False):
                from src.ai.ai_layer import AILayer
                
                ai = AILayer({
                    "detection_model": getattr(ai_cfg, "detection_model", "yolov8n.pt"),
                    "segmentation_model": getattr(ai_cfg, "segmentation_model", "facebook/sam-vit-base"),
                    "qa_model": getattr(ai_cfg, "qa_model", "gpt2")
                })
                
                _set_stage(registry, job_id, "EXPORTING", 98, "Running AI analysis…")
                
                ai_results = ai.analyze_scene(
                    job_id=job_id,
                    frames_dir=str(frames_dir),
                    metadata=metrics
                )
                
                # Update job with AI results
                _update_job(registry, job_id, 
                    ai_detections=ai_results.get("num_detections", 0),
                    ai_results=ai_results
                )
                print(f"[worker] AI analysis complete: {ai_results.get('num_detections', 0)} objects detected")
        except Exception as ai_err:
            print(f"[worker] AI analysis skipped: {ai_err}")

        # Evaluation metrics
        _set_stage(registry, job_id, "EXPORTING", 97, "Computing quality metrics…")
        try:
            psnr_vals, ssim_vals = [], []
            sample = train_cameras[:min(5, len(train_cameras))]
            for i, cam in enumerate(sample):
                pred = renderer_obj.render_torch(model, cam).clamp(0, 1)
                gt   = train_images[i].to(device).clamp(0, 1)
                psnr_vals.append(_compute_psnr(pred, gt))
                ssim_vals.append(_compute_ssim(pred, gt))
            metrics["psnr"] = round(sum(psnr_vals) / len(psnr_vals), 2)
            metrics["ssim"] = round(sum(ssim_vals) / len(ssim_vals), 4)
            print(f"[worker] PSNR={metrics['psnr']} dB  SSIM={metrics['ssim']}")
        except Exception as e:
            print(f"[worker] Metrics skipped: {e}")
            metrics["psnr"] = None
            metrics["ssim"] = None

        metrics["finished_at"]  = time.time()
        metrics["total_time_s"] = round(metrics["finished_at"] - metrics["started_at"], 1)
        _save_metrics(work_dir_path, metrics)

        rel = lambda p: str(Path(p).as_posix())
        _set_stage(
            registry, job_id, "READY", 100,
            f"Ready — {len(model):,} splats | "
            f"PSNR={metrics.get('psnr', 'n/a')} | "
            f"SSIM={metrics.get('ssim', 'n/a')}"
        )
        _update_job(
            registry, job_id,
            status="ready",
            ply_path=rel(ply_path),
            splat_path=rel(splat_path),
        )

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
        metrics["error"]       = str(exc)
        metrics["finished_at"] = time.time()
        _save_metrics(work_dir_path, metrics)
        raise