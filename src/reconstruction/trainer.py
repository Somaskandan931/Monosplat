"""
trainer.py — Core training loop for MonoSplat.

FIXES APPLIED (this version):
  [FIX-A] densify_grad_threshold default: 0.0005 (standard 3DGS).
  [FIX-B] densification_interval default: 500 (standard 3DGS).
  [FIX-C] lambda_dssim aligned to 0.2 (original 3DGS paper).
  [FIX-D] torch.cuda.empty_cache() every 500 iters.
  [FIX-E] NaN recovery logic.
  [FIX-F] absgrad / packed-mode gradient accumulation (gsplat compat).
  [FIX-G] SH degree schedule: increments active_sh_degree every 1000 iters.
  [FIX-H] GradScaler only active when device_type == "cuda".
  [FIX-I] _compute_loss: use dataset.view_index(viewpoint) O(1) instead of
          self.scene.views.index(viewpoint) which is O(N) per step and slows
          training significantly on large datasets (>100 views).
  [FIX-J] _render: reads camera pose from self.scene.images[viewpoint.image_id]
          (the normalized, authoritative pose) rather than viewpoint.tvec
          directly, ensuring normalization applied in train.py is actually used.
"""

import logging
import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class Trainer:
    """Trains a GaussianModel via differentiable splatting."""

    _DEFAULT_MAX_GAUSSIANS = 300000   # PATCH: 80000 → 300000

    def __init__(self, cfg: Dict, model: nn.Module, scene) -> None:
        self.cfg   = cfg
        self.model = model
        self.scene = scene

        train_cfg = cfg.get("training", {})

        # All defaults here are last-resort fallbacks ONLY — config.yaml is the
        # authoritative source.  Do NOT change these values to tune training;
        # edit configs/config.yaml instead.  The values below match the
        # config.yaml defaults exactly so that missing keys fall back
        # consistently rather than silently using stale magic numbers.
        self.iterations:             int   = train_cfg.get("iterations",             30000)
        self.densify_from_iter:      int   = train_cfg.get("densify_from_iter",      500)
        self.densify_until_iter:     int   = train_cfg.get("densify_until_iter",     15000)
        self.densification_interval: int   = train_cfg.get("densification_interval", 200)   # matches config.yaml default
        self.densify_grad_threshold: float = train_cfg.get("densify_grad_threshold", 0.0003) # matches config.yaml default
        self.max_gaussians:          int   = train_cfg.get("max_gaussians",          150000) # matches config.yaml default
        self.opacity_reset_interval: int   = train_cfg.get("opacity_reset_interval", 1000)
        self.lambda_dssim:           float = train_cfg.get("lambda_dssim",           0.2)   # FIX-C
        self.model_path:             str   = cfg.get("model_path", "outputs/gaussian")

        self._lr_cfg = {
            "position_lr_init":       train_cfg.get("position_lr_init",       0.00016),
            "position_lr_final":      train_cfg.get("position_lr_final",      0.0000016),
            "position_lr_delay_mult": train_cfg.get("position_lr_delay_mult", 0.01),
            # Default matches config.yaml training.position_lr_max_steps = 30000
            "position_lr_max_steps":  train_cfg.get("position_lr_max_steps",  30000),
            "feature_lr":             train_cfg.get("feature_lr",             0.0025),
            "opacity_lr":             train_cfg.get("opacity_lr",             0.05),
            "scaling_lr":             train_cfg.get("scaling_lr",             0.005),
            "rotation_lr":            train_cfg.get("rotation_lr",            0.001),
        }

        self.enable_densification: bool = True
        self.nan_counter:          int  = 0

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # FIX-H: GradScaler only for CUDA
        if self._device_type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self._last_good_ckpt: Optional[str] = None
        self._renderer = None
        self._run_tracker = None
        self._artifact_manager = None
        self._train_started_at = 0.0

        experiment_cfg = cfg.get("experiment", {})
        run_dir = experiment_cfg.get("run_dir")
        if run_dir:
            # core/ may not exist in this repo. Best-effort instrumentation only.
            try:
                from core.experiments.artifact_manager import ArtifactManager
                from core.experiments.run_tracker import RunTracker
                self._run_tracker = RunTracker(run_dir)
                self._artifact_manager = ArtifactManager(run_dir)
            except Exception:
                self._run_tracker = None
                self._artifact_manager = None


        try:
            import gsplat  # noqa: F401
        except ImportError:
            log.warning(
                "[Trainer] gsplat not found — software renderer active. "
                "Training will be slow and densification stats unavailable. "
                "Install with: pip install gsplat"
            )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        device = self._device_type
        self.model.to(device)
        self.model.train()
        self._setup_optimizer()

        viewpoint_stack = self.scene.get_train_cameras()
        self._train_started_at = time.time()

        for iteration in range(1, self.iterations + 1):
            viewpoint = _pick_viewpoint(viewpoint_stack)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device):
                render_pkg = self._render(viewpoint)
                loss       = self._compute_loss(render_pkg, viewpoint)

            if not torch.isfinite(loss):
                self.nan_counter += 1
                log.warning(
                    f"[Trainer] Invalid loss at iter {iteration}. "
                    f"NaN count = {self.nan_counter}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if self.nan_counter >= 3:
                    log.warning("[Trainer] Disabling densification due to instability")
                    self.enable_densification = False
                continue

            self.nan_counter = 0

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self._update_gradient_accum(render_pkg)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self._update_lr(iteration)

            # FIX-D: periodic VRAM cleanup; previews every 250 iters for faster debug feedback
            if iteration % 500 == 0:
                torch.cuda.empty_cache()
            if iteration % 250 == 0:
                self._save_preview(iteration)

            self._maybe_densify(iteration)

            if iteration % self.opacity_reset_interval == 0:
                if hasattr(self.model, "reset_opacity"):
                    self.model.reset_opacity()

            # FIX-G: SH degree schedule — activate one band every 1000 iters
            if iteration % 1000 == 0:
                if hasattr(self.model, "one_up_sh_degree"):
                    self.model.one_up_sh_degree()

            ckpt_iters = self.cfg.get("training", {}).get("checkpoint_iterations", [])
            # Colab-safe: checkpoint every 500 iters even between config milestones.
            # Prevents losing >8 min of GPU time on a runtime disconnect.
            if iteration in ckpt_iters or (iteration % 500 == 0 and iteration > 0):
                ckpt_path = self._save_checkpoint(iteration)
                self._last_good_ckpt = ckpt_path

            if iteration % 100 == 0:
                n_gaussians = self.model.get_xyz.shape[0]
                self._record_metrics(iteration, loss, render_pkg, viewpoint)
                log.info(
                    f"iter {iteration:>6}/{self.iterations}  "
                    f"loss={loss.item():.4f}  "
                    f"N={n_gaussians:,}  "
                    f"sh_deg={self.model.active_sh_degree}"
                )

        if self._run_tracker is not None:
            self._run_tracker.finalize(
                status="training_completed",
                extra={
                    "iterations": self.iterations,
                    "final_gaussians": int(self.model.get_xyz.shape[0]),
                    "last_checkpoint": self._last_good_ckpt,
                },
            )

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        lr = self._lr_cfg
        m  = self.model

        param_groups = [
            {"params": [m._xyz],           "lr": lr["position_lr_init"],  "name": "xyz"},
            {"params": [m._features_dc],   "lr": lr["feature_lr"],        "name": "f_dc"},
            {"params": [m._features_rest], "lr": lr["feature_lr"] / 20,  "name": "f_rest"},
            {"params": [m._opacities],     "lr": lr["opacity_lr"],        "name": "opacity"},
            {"params": [m._scales],        "lr": lr["scaling_lr"],        "name": "scaling"},
            {"params": [m._rotations],     "lr": lr["rotation_lr"],       "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        log.info("[Trainer] Optimizer initialised with %d parameter groups.", len(param_groups))

    def _update_lr(self, iteration: int) -> None:
        """Exponential log-linear LR decay (original 3DGS schedule)."""
        import math

        lr_cfg    = self._lr_cfg
        lr_init   = lr_cfg["position_lr_init"]
        lr_final  = lr_cfg["position_lr_final"]
        delay     = lr_cfg["position_lr_delay_mult"]
        max_steps = max(lr_cfg["position_lr_max_steps"], 1)

        t = iteration / max_steps
        if t >= 1.0:
            new_lr = lr_final
        else:
            log_lerp = (1.0 - t) * math.log(max(lr_init, 1e-15)) + t * math.log(max(lr_final, 1e-15))
            new_lr = math.exp(log_lerp) * (delay + (1.0 - delay) * t)

        for group in self.optimizer.param_groups:
            if group.get("name") == "xyz":
                group["lr"] = new_lr
                break


    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self, viewpoint) -> Dict:
        from renderer.renderer import GaussianRenderer
        from renderer.camera   import Camera

        if self._renderer is None:
            self._renderer = GaussianRenderer(device=self._device_type)

        # FIX-J: use the (possibly normalized) live image from dataset.images
        # rather than the raw viewpoint object, which may carry the original tvec.
        live_view = self.scene.images.get(viewpoint.image_id, viewpoint)
        cam_data  = self.scene.cameras[viewpoint.camera_id]

        camera = Camera.from_colmap(
            img_data=live_view,
            cam_data=cam_data,
            width=self.scene.width,
            height=self.scene.height,
        )

        result = self._renderer.render_torch(self.model, camera)
        if isinstance(result, tuple):
            rendered, meta = result
        else:
            rendered, meta = result, None

        return {
            "render":    rendered,
            "camera":    camera,
            "meta":      meta,
            "viewpoint": viewpoint,
        }

    # ------------------------------------------------------------------
    # Gradient accumulation (absgrad / packed-mode safe) — FIX-F
    # ------------------------------------------------------------------

    def _update_gradient_accum(self, render_pkg: Dict) -> None:
        meta = render_pkg.get("meta")
        if meta is None:
            return
        if not hasattr(self.model, "update_stats"):
            return

        radii_sparse = meta.get("radii")
        if radii_sparse is None:
            return

        means2d = meta.get("means2d")
        if means2d is None:
            return

        N      = self.model.get_xyz.shape[0]
        device = radii_sparse.device

        absgrad = getattr(means2d, "absgrad", None)
        if absgrad is not None:
            grad_sparse = absgrad.detach()
        elif means2d.grad is not None:
            grad_sparse = means2d.grad.detach()
        else:
            grad_sparse = torch.zeros(radii_sparse.shape[0], 2, device=device)

        gaussian_ids = meta.get("gaussian_ids")

        if gaussian_ids is not None:
            radii_dense = torch.zeros(N, dtype=torch.float32, device=device)
            grad_dense  = torch.zeros(N, 2, dtype=torch.float32, device=device)

            gids = gaussian_ids.view(-1).long().clamp(0, N - 1)

            radii_values = radii_sparse.view(-1).to(device=device, dtype=radii_dense.dtype)
            radii_dense.scatter_reduce_(0, gids, radii_values, reduce="amax", include_self=True)

            grad_norm_sparse = grad_sparse.view(-1, 2).to(dtype=torch.float32).norm(dim=-1)
            grad_dense_norm  = torch.zeros(N, dtype=torch.float32, device=device)
            grad_dense_norm.scatter_add_(0, gids, grad_norm_sparse)
        else:
            r = radii_sparse
            while r.dim() > 1:
                r = r.squeeze(0)
            radii_dense = r[:N]

            g = grad_sparse
            while g.dim() > 2:
                g = g.squeeze(0)
            grad_dense_norm = g[:N].norm(dim=-1)

        with torch.no_grad():
            self.model.update_stats_norm(radii_dense, grad_dense_norm)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, render_pkg: Dict, viewpoint) -> torch.Tensor:
        from reconstruction.loss import combined_loss

        rendered = render_pkg["render"]   # (3, H, W) float32 [0,1]

        # FIX-I: O(1) view → index lookup via cache
        idx      = self.scene.view_index(viewpoint)
        gt_item  = self.scene[idx]
        gt_image = gt_item["image"].to(rendered.device)

        if gt_image.shape != rendered.shape:
            import torch.nn.functional as F
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(rendered.shape[1], rendered.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Read lambda_lpips directly from config — no hardcoded fallback that
        # could override config.yaml.  The config_loader _DEFAULTS already
        # supplies 0.05 if the key is absent from config.yaml.
        lambda_lpips = self.cfg.get("training", {}).get("lambda_lpips", 0.05)
        return combined_loss(
            rendered, gt_image,
            lambda_ssim=self.lambda_dssim,
            lambda_lpips=lambda_lpips,
        )

    # ------------------------------------------------------------------
    # Checkpoint save / resume
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int) -> str:
        ckpt_dir  = Path(self.model_path) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / f"checkpoint_{iteration:06d}.ckpt")

        torch.save({
            "iteration": iteration,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler":    self.scaler.state_dict() if self.scaler else {},
            "max_log_scale": getattr(self.model, "_max_log_scale", None),
        }, ckpt_path)


        log.info(f"[Trainer] Checkpoint saved: {ckpt_path}")
        if self._artifact_manager is not None:
            self._artifact_manager.track(ckpt_path, kind="checkpoint", copy_to_run=True)
        drive_checkpoint_dir = self.cfg.get("runtime", {}).get("drive_checkpoint_dir")
        if drive_checkpoint_dir:
            # Best-effort mirror only; avoids dependency on missing core/.
            try:
                import shutil
                src = Path(ckpt_path)
                dst_root = Path(drive_checkpoint_dir)
                dst_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_root / src.name)
            except Exception:
                pass

        return ckpt_path

    def resume_from_checkpoint(self, ckpt_path: str) -> None:
        log.info(f"[Trainer] Resuming from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self._device_type)
        model_state = state.get("model") or state.get("model_state")
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError:
            log.info("[Trainer] Rebuilding Gaussian parameter tensors for checkpoint resume.")
            self._load_model_state_dynamic(model_state)
            self._setup_optimizer()
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])

        # Restore init scale ceiling so post-resume densification/cloning
        # obeys the same _max_log_scale used during initialise_from_pcd.
        max_log_scale = state.get("max_log_scale", None)
        if max_log_scale is not None:
            self.model._max_log_scale = float(max_log_scale)


        torch.cuda.empty_cache()
        with torch.no_grad():
            for p in self.model.parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    log.warning(f"[Trainer] Sanitising NaN/Inf in param shape={tuple(p.shape)}")
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

    def _load_model_state_dynamic(self, model_state: Dict) -> None:
        """Load checkpoint tensors whose Gaussian count differs from fresh init."""
        self.model._xyz = nn.Parameter(model_state["_xyz"].detach().to(self._device_type).float())
        self.model._features_dc = nn.Parameter(model_state["_features_dc"].detach().to(self._device_type).float())
        self.model._features_rest = nn.Parameter(model_state["_features_rest"].detach().to(self._device_type).float())
        self.model._opacities = nn.Parameter(model_state["_opacities"].detach().to(self._device_type).float())
        self.model._scales = nn.Parameter(model_state["_scales"].detach().to(self._device_type).float())
        self.model._rotations = nn.Parameter(model_state["_rotations"].detach().to(self._device_type).float())
        n = self.model._xyz.shape[0]
        device = self.model._xyz.device
        self.model.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        self.model.denom = torch.zeros(n, 1, device=device)
        self.model.max_radii2D = torch.zeros(n, device=device)

    # ------------------------------------------------------------------
    # Training preview saves (every 500 iters)
    # ------------------------------------------------------------------

    def _save_preview(self, iteration: int) -> None:
        """
        Render a preview image from the first training camera and save it.

        Lets you detect exploding geometry, opacity collapse, or bad poses
        early without waiting for the full training run to finish.
        Preview images are saved to <model_path>/previews/preview_XXXXXX.png.
        """
        try:
            viewpoint_stack = self.scene.get_train_cameras()
            if not viewpoint_stack:
                return

            # Always use the same camera (index 0) for consistent comparison
            test_camera = viewpoint_stack[0]

            with torch.no_grad():
                render_pkg = self._render(test_camera)
            rendered = render_pkg["render"]   # (3, H, W) float32

            preview_dir = Path(self.model_path) / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"preview_{iteration:06d}.png"

            # Convert (3, H, W) float32 → (H, W, 3) uint8 PNG
            import numpy as np
            from PIL import Image
            arr = (rendered.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(str(preview_path))

            log.info(f"[Trainer] Preview saved: {preview_path}")
            if self._artifact_manager is not None:
                self._artifact_manager.track(preview_path, kind="preview", copy_to_run=True)

        except Exception as exc:
            # Never crash the training loop on a preview failure
            log.warning(f"[Trainer] Preview save failed at iter {iteration}: {exc}")

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------

    def _maybe_densify(self, iteration: int) -> None:
        if not self.enable_densification:
            return
        if iteration < self.densify_from_iter:
            return
        if iteration > self.densify_until_iter:
            return
        if iteration % self.densification_interval != 0:
            return

        n_current = self.model.get_xyz.shape[0]
        if n_current >= self.max_gaussians:
            log.warning(
                f"Gaussian limit reached ({self.max_gaussians:,}). "
                f"Pruning lowest-opacity 20% at iter {iteration}."
            )
            # PATCH: prune 20% lowest-opacity instead of hard skip,
            # so densification can continue adding where it matters.
            with torch.no_grad():
                opacities = self.model.get_opacity.squeeze(-1)
                k_prune   = max(1, int(0.2 * n_current))
                _, prune_idx = torch.topk(opacities, k=k_prune, largest=False)
                prune_mask = torch.zeros(n_current, dtype=torch.bool, device=opacities.device)
                prune_mask[prune_idx] = True
            self.model._prune_points(prune_mask, self.optimizer)
            return

        # FIX: cameras_extent after scene normalization is often ~0.1 instead of ~1.0
        # because normalize_scene scales the *point cloud* to a unit sphere but the
        # camera centres remain clustered tightly.  Passing a tiny extent to
        # densify_and_prune sets the size-pruning threshold to
        #   percent_dense * extent = 0.01 * 0.1 = 0.001 world units,
        # which prunes virtually every Gaussian in the early densification window
        # and prevents the scene from ever building up detail (grey/foggy output).
        # Clamping to max(extent, 1.0) restores standard 3DGS behaviour.
        extent = max(self.scene.cameras_extent, 1.0)

        # Enable screen-size pruning only after a warmup window so large
        # screen-space floaters get removed once coverage has built up.
        # Screen-size pruning: only activate after enough coverage has built up,
        # AND use a conservative threshold (100 px, not 20 px).
        #
        # Root cause of the iter=2000 mass-prune observed in training logs:
        #   - max_screen=20 activated at first densify step past iter 1500
        #     (densify_from_iter=500 + 1000 = 1500 boundary)
        #   - max_radii2D had accumulated large values from 500 close-up renders
        #     with NO prior screen-size pruning to constrain them
        #   - result: big_vs flagged ~19,500 of 20,497 Gaussians → only 1,000
        #     survived (_prune_points min_keep floor), loss jumped to 0.65+,
        #     delta=+0 at every subsequent densify step → total training collapse
        #
        # Fix:
        #   1. Raise threshold 20 → 100 px (standard 3DGS value for this resolution)
        #   2. Defer first activation to densify_from_iter + 3000 so the scene has
        #      solid Gaussian coverage before any screen-size pruning ever fires.
        max_screen = 0
        if iteration > (self.densify_from_iter + 3000):
            max_screen = 100

        before = self.model.get_xyz.shape[0]

        self.model.densify_and_prune(
            max_grad=self.densify_grad_threshold,
            min_opacity=0.001,  # lowered from 0.005: after reset_opacity sets all
                                # opacities to 0.05, Gaussians decay for up to 500
                                # iters before the next densify fires. 0.005 was
                                # pruning recovering Gaussians that simply hadn't
                                # rebuilt opacity yet, causing the iter=4000/4500
                                # cascade collapse. 0.001 only removes true dead weight.
            extent=extent,
            max_screen_size=max_screen,
            optimizer=self.optimizer,
        )

        after = self.model.get_xyz.shape[0]
        log.info(
            "[DENSIFY] iter=%d  before=%d  after=%d  delta=%+d  extent=%.4f",
            iteration, before, after, after - before, extent,
        )


    # ------------------------------------------------------------------
    # Experiment metrics
    # ------------------------------------------------------------------

    def _record_metrics(self, iteration: int, loss: torch.Tensor, render_pkg: Dict, viewpoint) -> Dict:
        if self._run_tracker is None:
            return {}
        elapsed = time.time() - self._train_started_at
        iter_per_sec = iteration / max(elapsed, 1e-6)
        remaining = max(self.iterations - iteration, 0)
        eta = remaining / max(iter_per_sec, 1e-6)
        metrics = self._metric_values(render_pkg, viewpoint)
        entry = {
            "iteration": int(iteration),
            "loss": float(loss.detach().cpu().item()),
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "lpips": metrics.get("lpips"),
            "elapsed_time": elapsed,
            "eta": eta,
            "iteration_speed": iter_per_sec,
            "n_gaussians": int(self.model.get_xyz.shape[0]),
            "gpu_memory": self._gpu_memory(),
        }
        self._run_tracker.record(**entry)
        return entry

    def _metric_values(self, render_pkg: Dict, viewpoint) -> Dict:
        rendered = render_pkg["render"]
        idx = self.scene.view_index(viewpoint)
        gt_image = self.scene[idx]["image"].to(rendered.device)
        if gt_image.shape != rendered.shape:
            import torch.nn.functional as F
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(rendered.shape[1], rendered.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        try:
            from reconstruction.loss import lpips_metric, psnr_metric, ssim_metric
            psnr = float(psnr_metric(rendered, gt_image).detach().cpu().item())
            ssim = float((1.0 - ssim_metric(rendered, gt_image)).detach().cpu().item())
            lpips_value = float(lpips_metric(rendered, gt_image).detach().cpu().item())
            return {"psnr": psnr, "ssim": ssim, "lpips": lpips_value}
        except Exception as exc:
            log.warning("[Trainer] Metric computation failed at experiment record: %s", exc)
            return {"psnr": None, "ssim": None, "lpips": None}

    def _gpu_memory(self) -> Dict:
        if not torch.cuda.is_available():
            return {"available": False}
        device = self._device_type
        return {
            "available": True,
            "allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 4),
            "reserved_gb": round(torch.cuda.memory_reserved(device) / 1e9, 4),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 4),
        }


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _pick_viewpoint(stack):
    if not stack:
        raise RuntimeError("No training cameras available.")
    return random.choice(stack)