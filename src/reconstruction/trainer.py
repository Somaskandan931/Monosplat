"""
src/reconstruction/trainer.py
------------------------------
Training loop for MonoSplat — wired to diff-gaussian-rasterization.

KEY CHANGES vs previous version:
  [DG-1] Removed all gsplat-specific code:
          - no absgrad, no packed mode, no gaussian_ids scatter
          - no means2d.absgrad unscaling after scaler.backward()
  [DG-2] _update_gradient_accum now uses the original 3DGS mechanism:
          meta["viewspace_points"].grad  (standard autograd .grad)
          meta["visibility_filter"]      (bool mask, N)
          meta["radii"]                  (int, N)
         This is identical to gaussian-splatting/train.py lines 102-109.
  [DG-3] Removed autocast / GradScaler entirely.
          diff-gaussian-rasterization CUDA kernels do not support autocast.
          Training runs in float32 throughout — same as original repo.
  [DG-4] means2D.retain_grad() is called inside renderer._render_diff_gauss
          before rasterization — trainer just reads .grad after backward().

UNCHANGED vs previous version:
  - All densification logic (_maybe_densify, stage thresholds)
  - Optimizer setup and LR schedule
  - Checkpoint save/resume
  - Preview save logic
  - LPIPS warmup schedule
  - colab/train.py interface (Trainer(cfg, model, scene).train())
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

    _DEFAULT_MAX_GAUSSIANS = 300000

    def __init__(self, cfg: Dict, model: nn.Module, scene) -> None:
        self.cfg   = cfg
        self.model = model
        self.scene = scene

        train_cfg = cfg.get("training", {})

        self.iterations:             int   = train_cfg.get("iterations",             30000)
        self.densify_from_iter:      int   = train_cfg.get("densify_from_iter",      500)
        self.densify_until_iter:     int   = train_cfg.get("densify_until_iter",     15000)
        self.densification_interval: int   = train_cfg.get("densification_interval", 200)
        self.densify_grad_threshold: float = train_cfg.get("densify_grad_threshold", 0.0003)
        self.max_gaussians:          int   = train_cfg.get("max_gaussians",          150000)
        self.opacity_reset_interval: int   = train_cfg.get("opacity_reset_interval", 3000)
        self.lambda_dssim:           float = train_cfg.get("lambda_dssim",           0.2)
        self.model_path:             str   = cfg.get("model_path", "outputs/gaussian")

        self._lr_cfg = {
            "position_lr_init":       train_cfg.get("position_lr_init",       0.00016),
            "position_lr_final":      train_cfg.get("position_lr_final",      0.0000016),
            "position_lr_delay_mult": train_cfg.get("position_lr_delay_mult", 0.01),
            "position_lr_max_steps":  train_cfg.get("position_lr_max_steps",  30000),
            "feature_lr":             train_cfg.get("feature_lr",             0.0025),
            "opacity_lr":             train_cfg.get("opacity_lr",             0.05),
            "scaling_lr":             train_cfg.get("scaling_lr",             0.005),
            "rotation_lr":            train_cfg.get("rotation_lr",            0.001),
        }

        self.enable_densification: bool = True
        self.nan_counter:          int  = 0
        self._current_iter:        int  = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_good_ckpt: Optional[str] = None
        self._renderer = None
        self._train_started_at = 0.0

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
            self._current_iter = iteration
            viewpoint = _pick_viewpoint(viewpoint_stack)

            self.optimizer.zero_grad(set_to_none=True)

            # [DG-3] No autocast — diff-gaussian-rasterization requires float32.
            render_pkg = self._render(viewpoint)

            loss = self._compute_loss(render_pkg, viewpoint)

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

            # [DG-3] Plain backward — no scaler.
            loss.backward()

            # [DG-2] Read viewspace gradients AFTER backward.
            # means2D.retain_grad() was called in renderer before rasterization,
            # so means2D.grad is populated here with screen-space (x,y) gradients.
            self._update_gradient_accum(render_pkg)

            self.optimizer.step()
            self._update_lr(iteration)

            # Preview milestones — first meaningful preview at iter 500
            PREVIEW_MILESTONES = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 30000}
            if iteration in PREVIEW_MILESTONES or (iteration % 5000 == 0):
                self._save_preview(iteration)

            if iteration % 500 == 0:
                torch.cuda.empty_cache()

            self._maybe_densify(iteration)

            if iteration % self.opacity_reset_interval == 0:
                if hasattr(self.model, "reset_opacity"):
                    self.model.reset_opacity()

            # SH degree schedule — activate one band every 1000 iters
            if iteration % 1000 == 0:
                if hasattr(self.model, "one_up_sh_degree"):
                    self.model.one_up_sh_degree()

            ckpt_iters = self.cfg.get("training", {}).get("checkpoint_iterations", [])
            if iteration in ckpt_iters or (iteration % 500 == 0 and iteration > 0):
                ckpt_path = self._save_checkpoint(iteration)
                self._last_good_ckpt = ckpt_path

            if iteration % 100 == 0:
                n_gaussians = self.model.get_xyz.shape[0]
                log.info(
                    f"iter {iteration:>6}/{self.iterations}  "
                    f"loss={loss.item():.4f}  "
                    f"N={n_gaussians:,}  "
                    f"sh_deg={self.model.active_sh_degree}"
                )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        lr = self._lr_cfg
        m  = self.model

        param_groups = [
            {"params": [m._xyz],           "lr": lr["position_lr_init"],  "name": "xyz"},
            {"params": [m._features_dc],   "lr": lr["feature_lr"],        "name": "f_dc"},
            {"params": [m._features_rest], "lr": lr["feature_lr"] / 20,   "name": "f_rest"},
            {"params": [m._opacities],     "lr": lr["opacity_lr"],        "name": "opacity"},
            {"params": [m._scales],        "lr": lr["scaling_lr"],        "name": "scaling"},
            {"params": [m._rotations],     "lr": lr["rotation_lr"],       "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        log.info("[Trainer] Optimizer initialised with %d parameter groups.", len(param_groups))

    def _update_lr(self, iteration: int) -> None:
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
    # Gradient accumulation — [DG-2] original 3DGS mechanism
    # ------------------------------------------------------------------

    def _update_gradient_accum(self, render_pkg: Dict) -> None:
        """
        Accumulate screen-space gradient norms for densification.

        This is the direct equivalent of gaussian-splatting/train.py lines 102-109:

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )
            gaussians.add_densification_stats(
                viewspace_point_tensor, visibility_filter
            )

        viewspace_point_tensor.grad is populated by diff-gaussian-rasterization's
        CUDA backward kernel writing into means2D.grad via standard autograd.
        No absgrad hook, no packed-mode scatter, no gaussian_ids needed.
        """
        meta = render_pkg.get("meta")
        if meta is None:
            return

        viewspace_points   = meta.get("viewspace_points")
        visibility_filter  = meta.get("visibility_filter")
        radii              = meta.get("radii")

        if viewspace_points is None or visibility_filter is None or radii is None:
            return

        if viewspace_points.grad is None:
            # backward() has not run yet or means2D wasn't used in graph
            return

        if not hasattr(self.model, "update_stats"):
            return

        N      = self.model.get_xyz.shape[0]
        device = radii.device

        # Update max screen radii for visible Gaussians
        radii_dense = radii.float().to(device)          # (N,)
        vis         = visibility_filter.to(device)       # (N,) bool

        # Gradient norm: L2 of (∂loss/∂x_screen, ∂loss/∂y_screen) per Gaussian
        grad = viewspace_points.grad.detach()            # (N, 3) or (N, 2)
        grad_norm = grad[:N, :2].norm(dim=-1)            # (N,) — only xy screen coords

        with torch.no_grad():
            self.model.update_stats_norm(radii_dense, grad_norm)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, render_pkg: Dict, viewpoint) -> torch.Tensor:
        from reconstruction.loss import combined_loss

        rendered = render_pkg["render"]   # (3, H, W) float32 [0,1]

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

        # LPIPS warmup: 0 until iter 1000, linear ramp to full by iter 3000
        lambda_lpips_cfg = self.cfg.get("training", {}).get("lambda_lpips", 0.05)
        if self._current_iter < 1000:
            lambda_lpips = 0.0
        elif self._current_iter < 3000:
            ramp = (self._current_iter - 1000) / 2000.0
            lambda_lpips = lambda_lpips_cfg * ramp
        else:
            lambda_lpips = lambda_lpips_cfg

        return combined_loss(
            rendered, gt_image,
            lambda_ssim=self.lambda_dssim,
            lambda_lpips=lambda_lpips,
        )

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
            log.warning(f"Gaussian limit ({self.max_gaussians:,}). Pruning 20% at iter {iteration}.")
            with torch.no_grad():
                opacities = self.model.get_opacity.squeeze(-1)
                k_prune   = max(1, int(0.2 * n_current))
                _, prune_idx = torch.topk(opacities, k=k_prune, largest=False)
                prune_mask = torch.zeros(n_current, dtype=torch.bool, device=opacities.device)
                prune_mask[prune_idx] = True
            self.model._prune_points(prune_mask, self.optimizer)
            return

        # Grad diagnostic
        with torch.no_grad():
            if hasattr(self.model, "xyz_gradient_accum") and hasattr(self.model, "denom"):
                _grads_diag = self.model.xyz_gradient_accum / self.model.denom.clamp_min(1)
                _grads_diag[_grads_diag.isnan()] = 0.0
                log.info(
                    "[DENSIFY-DIAG] grad_accum  max=%.6f  mean=%.6f",
                    _grads_diag.max().item(), _grads_diag.mean().item(),
                )

        # Opacity diagnostic
        with torch.no_grad():
            opacity = self.model.get_opacity
            log.info(
                "[DEBUG] opacity min=%.6f  mean=%.6f  max=%.6f",
                opacity.min().item(), opacity.mean().item(), opacity.max().item(),
            )

        extent = max(self.scene.cameras_extent, 1.0)

        # [THRESH-FIX-1] 3-stage densification now derives grad_thresh from
        # self.densify_grad_threshold (which comes from config.yaml) rather than
        # using hardcoded values (0.00003 / 0.00005) that ignored the config.
        # Stage multipliers: stabilize=0.1x, controlled=0.25x, normal=1.0x of config.
        # This ensures config.yaml densify_grad_threshold is actually respected.
        cfg_thresh = self.densify_grad_threshold
        if iteration < 2000:
            min_opacity = 0.0
            grad_thresh = cfg_thresh * 0.1     # very permissive warmup
            max_screen  = 0
        elif iteration < 15000:
            min_opacity = 0.0001
            grad_thresh = cfg_thresh * 0.25    # controlled growth
            max_screen  = 0
        else:
            min_opacity = 0.0005
            grad_thresh = cfg_thresh           # full config value
            max_screen  = 0   # big_vs disabled — world-space prune sufficient

        log.info(
            "[THRESH-DIAG] CONFIG=%.5f  USED=%.5f  iter=%d",
            cfg_thresh, grad_thresh, iteration,
        )

        before = self.model.get_xyz.shape[0]
        self.model.densify_and_prune(
            max_grad=grad_thresh,
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen,
            optimizer=self.optimizer,
        )
        after = self.model.get_xyz.shape[0]

        if after < 10000:
            log.critical(
                "[CRITICAL] Gaussian collapse at iter %d: %d remaining.",
                iteration, after,
            )
            return

        log.info(
            "[DENSIFY] iter=%d  before=%d  after=%d  delta=%+d  "
            "stage=%s  min_opacity=%.4f  grad_thresh=%.5f  extent=%.4f",
            iteration, before, after, after - before,
            "stabilize" if iteration < 2000 else ("controlled" if iteration < 15000 else "normal"),
            min_opacity, grad_thresh, extent,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int) -> str:
        ckpt_dir  = Path(self.model_path) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / f"checkpoint_{iteration:06d}.ckpt")
        torch.save({
            "iteration":     iteration,
            "model":         self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "max_log_scale": getattr(self.model, "_max_log_scale", None),
        }, ckpt_path)
        log.info(f"[Trainer] Checkpoint saved: {ckpt_path}")
        drive_checkpoint_dir = self.cfg.get("runtime", {}).get("drive_checkpoint_dir")
        if drive_checkpoint_dir:
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
            log.info("[Trainer] Rebuilding Gaussian tensors for resume.")
            self._load_model_state_dynamic(model_state)
            self._setup_optimizer()
        self.optimizer.load_state_dict(state["optimizer"])
        max_log_scale = state.get("max_log_scale")
        if max_log_scale is not None:
            self.model._max_log_scale = float(max_log_scale)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for p in self.model.parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

    def _load_model_state_dynamic(self, model_state: Dict) -> None:
        self.model._xyz           = nn.Parameter(model_state["_xyz"].detach().to(self._device_type).float())
        self.model._features_dc   = nn.Parameter(model_state["_features_dc"].detach().to(self._device_type).float())
        self.model._features_rest = nn.Parameter(model_state["_features_rest"].detach().to(self._device_type).float())
        self.model._opacities     = nn.Parameter(model_state["_opacities"].detach().to(self._device_type).float())
        self.model._scales        = nn.Parameter(model_state["_scales"].detach().to(self._device_type).float())
        self.model._rotations     = nn.Parameter(model_state["_rotations"].detach().to(self._device_type).float())
        n = self.model._xyz.shape[0]
        device = self.model._xyz.device
        self.model.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        self.model.denom              = torch.zeros(n, 1, device=device)
        self.model.max_radii2D        = torch.zeros(n,    device=device)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _save_preview(self, iteration: int) -> None:
        try:
            viewpoint_stack = self.scene.get_train_cameras()
            if not viewpoint_stack:
                return
            test_camera = viewpoint_stack[0]
            with torch.no_grad():
                render_pkg = self._render(test_camera)
            rendered = render_pkg["render"]
            preview_dir = Path(self.model_path) / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"preview_{iteration:06d}.png"
            import numpy as np
            from PIL import Image
            arr = (rendered.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(str(preview_path))
            log.info(f"[Trainer] Preview saved: {preview_path}")
        except Exception as exc:
            log.warning(f"[Trainer] Preview failed at iter {iteration}: {exc}")

    # ------------------------------------------------------------------
    # Experiment metrics
    # ------------------------------------------------------------------

    def _metric_values(self, render_pkg: Dict, viewpoint) -> Dict:
        rendered = render_pkg["render"]
        idx = self.scene.view_index(viewpoint)
        gt_image = self.scene[idx]["image"].to(rendered.device)
        if gt_image.shape != rendered.shape:
            import torch.nn.functional as F
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=(rendered.shape[1], rendered.shape[2]),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        try:
            from reconstruction.loss import lpips_metric, psnr_metric, ssim_metric
            return {
                "psnr":  float(psnr_metric(rendered, gt_image).detach().cpu()),
                "ssim":  float((1.0 - ssim_metric(rendered, gt_image)).detach().cpu()),
                "lpips": float(lpips_metric(rendered, gt_image).detach().cpu()),
            }
        except Exception as exc:
            log.warning("[Trainer] Metric computation failed: %s", exc)
            return {"psnr": None, "ssim": None, "lpips": None}

    def _gpu_memory(self) -> Dict:
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available":       True,
            "allocated_gb":    round(torch.cuda.memory_allocated() / 1e9, 4),
            "reserved_gb":     round(torch.cuda.memory_reserved()  / 1e9, 4),
            "max_allocated_gb":round(torch.cuda.max_memory_allocated() / 1e9, 4),
        }


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _pick_viewpoint(stack):
    if not stack:
        raise RuntimeError("No training cameras available.")
    return random.choice(stack)