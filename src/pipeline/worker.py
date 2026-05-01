"""
worker.py
Async pipeline worker for MonoSplat.

Product pipeline behavior:
    - Adaptive FPS passed to extract_from_video (duration-based)
    - max_frames=600
    - blur_threshold=80 (lenient)
    - exhaustive_matcher is now the default in colmap_runner
    - GPU flags passed to colmap
    - Auto-retry system: if registration < 50%, retry with relaxed thresholds
    - Reject bad videos early: motion < 1.0 → warning + reject
    - Input validation: too few frames, low texture, no parallax
    - Gaussian training iterations boosted to 60000 when points < 10000

Split workflow:
    Desktop (CPU) → steps 1-2, stop with "ready_for_colab"
    Colab (GPU)   → step 3 training, step 4 export
"""

import json
import time
import traceback
from pathlib import Path
from typing import Callable, Optional


STAGES = [
    ("EXTRACTING",  5,  "Extracting frames from video"),
    ("COLMAP",     35,  "Running COLMAP pose estimation"),
    ("TRAINING",   85,  "Training Gaussian Splat model (GPU required)"),
    ("EXPORTING",  95,  "Exporting .ply and .splat files"),
    ("READY",     100,  "Done"),
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
    import json

    def _convert(o):
        # Handle NumPy types
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)

        # Handle PyTorch tensors
        if isinstance(o, torch.Tensor):
            return o.item()

        # Handle Path objects
        if isinstance(o, Path):
            return str(o)

        # Fallback
        return str(o)

    path = work_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_convert)

    print(f"[worker] Metrics saved → {path}")


def run_cmd_silent(cmd: list, step_name: str) -> int:
    """Run a subprocess, print output, return exit code (never raises)."""
    import subprocess
    print(f"\n[worker] ▶  {step_name}")
    print("  " + " ".join(str(c) for c in cmd))
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
    """Delegate to loss.py canonical implementation to avoid duplication."""
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
        mu1 = pred.mean().item()
        mu2 = gt.mean().item()
        s1  = pred.std().item()
        s2  = gt.std().item()
        s12 = ((pred - mu1) * (gt - mu2)).mean().item()
        c1, c2 = 0.01**2, 0.03**2
        num = (2*mu1*mu2 + c1) * (2*s12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)
        return num / (den + 1e-8)


def _count_registered(text_model: Path) -> tuple:
    """
    Returns (registered, total_in_file) from images.txt.

    images.txt format (COLMAP text):
        # comment lines
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME   ← image header line (9+ tokens)
        POINTS2D[] as (X, Y, POINT3D_ID) ...            ← keypoint line (variable tokens)

    The two lines alternate, so we only count odd-numbered non-comment lines
    (i.e., every other line starting from the first data line) to avoid
    double-counting image headers and their keypoint continuation lines.
    """
    images_txt = text_model / "images.txt"
    if not images_txt.exists():
        return 0, 0
    registered = 0
    data_line_idx = 0  # counts only non-comment, non-empty lines
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Image header lines are at even data_line_idx (0, 2, 4 …)
            # Keypoint lines are at odd data_line_idx (1, 3, 5 …) — skip them
            if data_line_idx % 2 == 0:
                registered += 1
            data_line_idx += 1
    return registered, registered


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

    from src.utils.env_detect import has_torch_gpu
    has_gpu = has_torch_gpu()
    print(f"[worker] GPU available: {has_gpu}")

    try:
        # ----------------------------------------------------------------
        # STAGE 1 — Frame extraction
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXTRACTING", 5, "Extracting frames with FFmpeg…")

        from src.preprocessing.extract_frames import extract_from_video, estimate_motion

        # fps=None → adaptive by duration (item 1)
        n_frames = extract_from_video(
            str(input_path),
            output_dir=str(frames_dir),
            fps=None,           # adaptive FPS
            max_frames=600,     # item 8: increased max frames
            blur_threshold=80.0,
            adaptive_sampling=False,
        )
        metrics["num_frames"] = n_frames

        # Input validation (item 7)
        if n_frames < 10:
            raise RuntimeError(
                f"Too few frames extracted ({n_frames}). "
                "Record at least 5 seconds of smooth footage around the object."
            )

        # Low texture validation — estimate avg features from extracted frames
        try:
            from src.preprocessing.extract_frames import avg_features_estimate
            avg_feat = avg_features_estimate(str(frames_dir))
            metrics["avg_features"] = round(avg_feat, 1)
            print(f"[worker] Avg features per frame: {avg_feat:.0f}")
            if avg_feat < 500:
                print(
                    f"[worker] WARNING: low texture detected (avg_features={avg_feat:.0f}). "
                    "COLMAP may fail; continuing so usable videos are not rejected early."
                )
                _update_job(
                    registry, job_id,
                    message=(
                        f"Low texture warning: avg_features={avg_feat:.0f}. "
                        "Continuing with COLMAP."
                    ),
                )
        except ImportError:
            pass

        # Auto re-sample if still too few frames after extraction
        if n_frames < 20:
            # fps=None (adaptive) produced too few frames — force a higher fixed fps.
            # Repeating the same adaptive call would yield the same result.
            fps_retry = 5
            print(f"[worker] ⚠  Only {n_frames} frames — triggering re-extraction at fps={fps_retry}…")
            _set_stage(registry, job_id, "EXTRACTING", 12, f"Too few frames ({n_frames}) — re-extracting at fps={fps_retry}…")
            import shutil as _shutil
            _shutil.rmtree(str(frames_dir), ignore_errors=True)
            frames_dir.mkdir(parents=True, exist_ok=True)
            n_frames = extract_from_video(
                str(input_path),
                output_dir=str(frames_dir),
                fps=fps_retry,
                max_frames=600,
                blur_threshold=80.0,
                adaptive_sampling=False,
            )
            metrics["num_frames_retry"] = n_frames
            print(f"[worker] Re-extraction result: {n_frames} frames")

        # Motion validation (item 14 / reject bad videos early)
        motion = estimate_motion(str(frames_dir))
        metrics["motion_score"] = round(motion, 3)
        print(f"[worker] Motion score: {motion:.2f}")

        if motion < 1.0:
            raise RuntimeError(
                f"No camera movement detected (motion={motion:.2f}). "
                "Walk around the object slowly. Do NOT rotate the object in place."
            )

        _update_job(registry, job_id, num_images=n_frames)
        _set_stage(registry, job_id, "EXTRACTING", 20, f"Extracted {n_frames} frames ✓ (motion={motion:.1f})")

        # ----------------------------------------------------------------
        # STAGE 2 — COLMAP (exhaustive_matcher, GPU enabled)
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
            use_gpu=True,   # GPU enabled (item 10)
            quality="medium",
            on_progress=colmap_progress,
        )

        text_model = colmap_dir / "sparse_text"

        # ---- Auto-retry if registration is poor (item 13) ----------------
        registered, _ = _count_registered(text_model)
        registration_ratio = registered / max(n_frames, 1)
        metrics["registered_frames"] = registered
        metrics["registration_ratio"] = round(registration_ratio, 3)

        if registration_ratio < 0.5 and text_model.exists():
            print(
                f"[worker] [AUTO-FIX] Registration only {registration_ratio*100:.0f}% "
                f"({registered}/{n_frames}) — retrying with relaxed thresholds…"
            )
            _set_stage(
                registry, job_id, "COLMAP", 40,
                f"Auto-retry: low registration ({registration_ratio*100:.0f}%) — relaxing thresholds…"
            )

            # Wipe old colmap output and retry with low quality thresholds
            import shutil
            if colmap_dir.exists():
                shutil.rmtree(colmap_dir)
            colmap_dir.mkdir(parents=True, exist_ok=True)

            run_colmap(
                image_dir=str(frames_dir),
                output_dir=str(colmap_dir),
                colmap_binary=colmap_binary,
                use_gpu=True,
                quality="low",   # relaxed thresholds
                on_progress=colmap_progress,
            )

            registered, _ = _count_registered(text_model)
            registration_ratio = registered / max(n_frames, 1)
            metrics["registered_frames_retry"] = registered
            print(f"[worker] After retry: {registered}/{n_frames} registered ({registration_ratio*100:.0f}%)")

        if not text_model.exists() or not any(text_model.iterdir()):
            raise RuntimeError(
                "COLMAP produced no model. "
                "Ensure 60%+ overlap and good lighting. "
                "Walk around the object slowly with it centred in frame."
            )

        metrics["colmap_time_s"] = round(time.time() - t_colmap, 1)
        _set_stage(
            registry, job_id, "COLMAP", 55,
            f"COLMAP complete — {registered}/{n_frames} frames registered ✓"
        )

        # ----------------------------------------------------------------
        # STAGE 2b — Dense reconstruction (WOW FACTOR)
        # image_undistorter → patch_match_stereo → stereo_fusion
        # Produces a dense point cloud alongside the sparse one.
        # Only runs if we have enough registered frames to be worthwhile.
        # ----------------------------------------------------------------
        dense_dir = work_dir_path / "colmap" / "dense"
        if registered >= 10:
            _set_stage(registry, job_id, "COLMAP", 57, "Running dense reconstruction…")
            try:
                sparse_0 = colmap_dir / "sparse" / "0"
                dense_dir.mkdir(parents=True, exist_ok=True)

                # Step 1: undistort images for MVS
                undistort_cmd = [
                    colmap_binary, "image_undistorter",
                    "--image_path",      str(frames_dir),
                    "--input_path",      str(sparse_0),
                    "--output_path",     str(dense_dir),
                    "--output_type",     "COLMAP",
                    "--max_image_size",  "1000",
                ]
                rc_u = run_cmd_silent(undistort_cmd, "Dense: image_undistorter")

                if rc_u == 0:
                    # Step 2: patch match stereo (depth maps)
                    pms_cmd = [
                        colmap_binary, "patch_match_stereo",
                        "--workspace_path",   str(dense_dir),
                        "--workspace_format", "COLMAP",
                        "--PatchMatchStereo.geom_consistency", "1",
                    ]
                    rc_p = run_cmd_silent(pms_cmd, "Dense: patch_match_stereo")

                    if rc_p == 0:
                        # Step 3: fuse depth maps into dense point cloud
                        dense_ply = dense_dir / "fused.ply"
                        fusion_cmd = [
                            colmap_binary, "stereo_fusion",
                            "--workspace_path",   str(dense_dir),
                            "--workspace_format", "COLMAP",
                            "--input_type",       "geometric",
                            "--output_path",      str(dense_ply),
                        ]
                        rc_f = run_cmd_silent(fusion_cmd, "Dense: stereo_fusion")

                        if rc_f == 0 and dense_ply.exists():
                            size_mb = dense_ply.stat().st_size / 1e6
                            print(f"[worker] ✅ Dense reconstruction complete → {dense_ply} ({size_mb:.1f} MB)")
                            metrics["dense_ply"] = str(dense_ply)
                            _update_job(registry, job_id, message=f"Dense reconstruction complete ({size_mb:.1f} MB)")
                        else:
                            print("[worker] ⚠  stereo_fusion failed — skipping dense (sparse still used)")
                    else:
                        print("[worker] ⚠  patch_match_stereo failed — skipping dense")
                else:
                    print("[worker] ⚠  image_undistorter failed — skipping dense")

            except Exception as e:
                print(f"[worker] ⚠  Dense reconstruction skipped: {e}")
        else:
            print(f"[worker] Skipping dense reconstruction — only {registered} frames registered (need ≥10)")

        # ----------------------------------------------------------------
        # STAGE 3 — Training (GPU only)
        # ----------------------------------------------------------------
        if not has_gpu:
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
            metrics["colmap_path"]    = str(colmap_dir)
            metrics["frames_path"]    = str(frames_dir)
            metrics["ready_for_colab"] = True
            _save_metrics(work_dir_path, metrics)
            return

        _set_stage(registry, job_id, "TRAINING", 60, "Starting Gaussian Splat training on GPU…")
        t_train = time.time()

        from src.utils.config_loader import load_config
        cfg = load_config(config_path)

        device = "cuda"
        print(f"[worker] Training device: {device}")

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

        from src.reconstruction.gaussian_model import GaussianModel
        from src.reconstruction.trainer import GaussianTrainer
        from src.renderer.renderer import GaussianRenderer

        xyzs = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
        rgbs = np.array([p.rgb for p in points3d.values()], dtype=np.float32) / 255.0

        model = GaussianModel(sh_degree=cfg.renderer.sh_degree)
        model.create_from_points(xyzs, rgbs)

        # Item 5 (Gaussian): boost iterations when point cloud is sparse
        num_points = len(xyzs)
        if num_points < 10000:
            cfg.training.iterations = max(cfg.training.iterations, 60000)
            print(f"[worker] Sparse point cloud ({num_points} pts) — boosting iterations to {cfg.training.iterations}")

        cfg.training.output_dir     = str(model_dir / "gaussian")
        cfg.training.checkpoint_dir = str(model_dir / "checkpoints")

        renderer_obj = GaussianRenderer(
            width=cfg.viewer.window_width,
            height=cfg.viewer.window_height,
            bg_color=cfg.renderer.background_color,
            device=device,
        )

        trainer = GaussianTrainer(model, renderer_obj.render_torch, train_cameras, train_images, cfg)

        total_iters  = cfg.training.iterations
        _last_pct    = [55]

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
        metrics["num_gaussians"]   = len(model)
        _update_job(registry, job_id, num_gaussians=len(model))
        _set_stage(registry, job_id, "TRAINING", 85, f"Training complete — {len(model):,} Gaussians ✓")

        # ----------------------------------------------------------------
        # STAGE 4 — Export
        # ----------------------------------------------------------------
        _set_stage(registry, job_id, "EXPORTING", 90, "Exporting .ply and .splat…")

        from src.utils.io_utils import save_ply, save_splat
        ply_path   = model_dir / "gaussian" / f"{job_id}.ply"
        splat_path = model_dir / "gaussian" / f"{job_id}.splat"

        save_ply(str(ply_path), model.get_state())
        save_splat(str(splat_path), model.get_state())

        _update_job(registry, job_id, ply_path=str(ply_path), splat_path=str(splat_path))
        _set_stage(registry, job_id, "EXPORTING", 95, "Files exported ✓")

        # ---- Evaluation metrics ------------------------------------------
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
        # Note: metrics dict is persisted to disk via _save_metrics above.
        # Do NOT pass it to update_job — ModelJob has no metrics field and
        # setattr would add it dynamically but it won't survive registry serialisation.

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
        metrics["error"]       = str(exc)
        metrics["finished_at"] = time.time()
        _save_metrics(work_dir_path, metrics)
        raise
