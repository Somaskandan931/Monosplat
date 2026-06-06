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
        self.opacity_reset_interval: int   = train_cfg.get("opacity_reset_interval", 3000)  # matches config.yaml default; safe now that reset_opacity uses jitter
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
        self._train_started_at = 0.0


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
            self._current_iter = iteration   # [BLUR-FIX-2] expose to _compute_loss for LPIPS warmup

            self.optimizer.zero_grad(set_to_none=True)

            # Render OUTSIDE autocast — means2d must stay fp32 so that gsplat's
            # absgrad hook attaches to the correct tensor node.  Inside autocast,
            # means2d is cast to fp16 and absgrad ends up on the fp16 copy, which
            # getattr(means2d, "absgrad") cannot find → grad_sparse = zeros →
            # xyz_gradient_accum never grows → delta=+0 every densification step.
            render_pkg = self._render(viewpoint)

            # Retain grad on means2d as a fallback for non-absgrad gsplat builds
            meta = render_pkg.get("meta")
            if meta is not None:
                means2d = meta.get("means2d")
                if means2d is not None and means2d.requires_grad:
                    means2d.retain_grad()

            with torch.amp.autocast(device):
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

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Fix: unscale means2d.absgrad before reading it in _update_gradient_accum.
            # scaler.scale(loss).backward() inflates ALL gradients (including absgrad)
            # by the scaler factor (typically 65536).  We must undo that scaling before
            # accumulating into xyz_gradient_accum, otherwise the grad threshold
            # (0.0002) is never meaningfully compared against inflated values — the
            # accumulator grows but comparison semantics are wrong.
            # scaler.unscale_(optimizer) only unscales optimizer-tracked params, NOT
            # intermediate tensors like means2d, so we do it manually here.
            if self.scaler is not None:
                inv_scale = 1.0 / (self.scaler.get_scale() + 1e-8)
                meta = render_pkg.get("meta")
                if meta is not None:
                    means2d = meta.get("means2d")
                    if means2d is not None:
                        ab = getattr(means2d, "absgrad", None)
                        if ab is not None:
                            means2d.absgrad = ab * inv_scale
                        elif means2d.grad is not None:
                            means2d.grad.mul_(inv_scale)

            self._update_gradient_accum(render_pkg)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self._update_lr(iteration)

            # [BLUR-FIX-5] Preview schedule: skip early previews (always blurry before
            # densification kicks in at iter 500). Save at meaningful milestones instead:
            #   iter 250 is guaranteed fog — densification hasn't run yet, N=50K frozen.
            #   First useful preview is iter 1000 (2 densification cycles completed).
            PREVIEW_MILESTONES = {500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 30000}
            if iteration in PREVIEW_MILESTONES or (iteration % 5000 == 0):
                self._save_preview(iteration)

            # FIX-D: periodic VRAM cleanup
            if iteration % 500 == 0:
                torch.cuda.empty_cache()

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
                log.info(
                    f"iter {iteration:>6}/{self.iterations}  "
                    f"loss={loss.item():.4f}  "
                    f"N={n_gaussians:,}  "
                    f"sh_deg={self.model.active_sh_degree}"
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

        # [BLUR-FIX-2] LPIPS warmup: don't apply LPIPS until iter 3000.
        #
        # LPIPS is a high-level perceptual loss that operates on VGG feature
        # maps.  Before the scene has geometric structure (iter < 3000), LPIPS
        # pushes Gaussians to form large blobs matching perceptual features
        # rather than actual geometry.  This fights L1+SSIM and causes the
        # loss to go UP between iters 200–400 (visible in the training log).
        #
        # Warmup schedule:
        #   iter < 1000  : no LPIPS (pure L1 + SSIM — geometry stabilization)
        #   iter < 3000  : LPIPS ramped up linearly from 0 → lambda_lpips
        #   iter >= 3000 : full LPIPS weight from config
        #
        # This matches the strategy used in Scaffold-GS and Mip-Splatting.
        lambda_lpips_cfg = self.cfg.get("training", {}).get("lambda_lpips", 0.05)
        current_iter = getattr(self, "_current_iter", self.iterations)  # set by train loop

        lpips_warmup_start = 1000
        lpips_warmup_end   = 3000
        if current_iter < lpips_warmup_start:
            lambda_lpips = 0.0
        elif current_iter < lpips_warmup_end:
            ramp = (current_iter - lpips_warmup_start) / (lpips_warmup_end - lpips_warmup_start)
            lambda_lpips = lambda_lpips_cfg * ramp
        else:
            lambda_lpips = lambda_lpips_cfg

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
            with torch.no_grad():
                opacities = self.model.get_opacity.squeeze(-1)
                k_prune   = max(1, int(0.2 * n_current))
                _, prune_idx = torch.topk(opacities, k=k_prune, largest=False)
                prune_mask = torch.zeros(n_current, dtype=torch.bool, device=opacities.device)
                prune_mask[prune_idx] = True
            self.model._prune_points(prune_mask, self.optimizer)
            return

        # ── Opacity debug visibility ──────────────────────────────────────────
        # Shows distribution before each densify step — useful for diagnosing
        # collapse (mean near 0) or saturation (mean near 1).
        with torch.no_grad():
            opacity = self.model.get_opacity
            log.info(
                "[DEBUG] opacity min=%.6f  mean=%.6f  max=%.6f",
                opacity.min().item(),
                opacity.mean().item(),
                opacity.max().item(),
            )

        # ── cameras_extent clamp ──────────────────────────────────────────────
        # After normalize_scene cameras_extent can be ~0.1, making the
        # size-pruning threshold 10× too tight.  Floor to 1.0 restores
        # standard 3DGS screen-space pruning behaviour.
        extent = max(self.scene.cameras_extent, 1.0)

        # ── Stage-based training logic ────────────────────────────────────────
        #
        # Stage 1 (iter < 2000): Stabilization — no pruning, low grad threshold.
        #   Scene is still learning structure; aggressive pruning here causes
        #   Gaussian collapse before the model has enough signal to recover.
        #
        # Stage 2 (2000 ≤ iter < 6000): Controlled pruning — gentle opacity
        #   floor.  Coverage is established; light cleanup is safe.
        #
        # Stage 3 (iter ≥ 6000): Normal pruning — standard 3DGS thresholds.
        #   Model is mature; prune aggressively to remove floaters.
        #
        # [BLUR-FIX-6] Stage 1 grad_thresh lowered 0.0002 → 0.00015.
        # With 150K init Gaussians on a normalized scene, the mean gradient
        # magnitude per visible Gaussian is lower (signal shared across more
        # primitives). A tighter threshold ensures enough clone/split events
        # happen per densification step to build coverage.
        if iteration < 2000:
            min_opacity = 0.0       # no opacity pruning in stabilization stage
            grad_thresh = 0.00015   # [BLUR-FIX-6] was 0.0002 — more aggressive early cloning
            max_screen  = 0         # no screen-size pruning yet
        elif iteration < 6000:
            min_opacity = 0.0001    # very gentle floor
            grad_thresh = 0.0002
            max_screen  = 0
        else:
            min_opacity = 0.0005    # standard-ish floor
            grad_thresh = 0.0003
            max_screen  = 20 if iteration > (self.densify_from_iter + 1000) else 0

        before = self.model.get_xyz.shape[0]

        self.model.densify_and_prune(
            max_grad=grad_thresh,
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen,
            optimizer=self.optimizer,
        )

        after = self.model.get_xyz.shape[0]

        # ── Critical safety guard ─────────────────────────────────────────────
        # If pruning wiped out too many Gaussians, log the collapse and skip
        # (the prune already happened, but future steps will re-densify).
        # The floor is raised to 10000 in _prune_points in gaussian_model.py
        # so in practice this block fires only if that guard was bypassed.
        MIN_GAUSSIANS = 10000
        if after < MIN_GAUSSIANS:
            log.critical(
                "[CRITICAL] Gaussian collapse detected at iter %d: %d Gaussians remaining. "
                "Densification disabled for this step. Check opacity distribution above.",
                iteration, after,
            )
            # Nothing more to do — the guard in _prune_points should have
            # prevented this, but log it prominently so it's visible.
            return

        log.info(
            "[DENSIFY] iter=%d  before=%d  after=%d  delta=%+d  "
            "stage=%s  min_opacity=%.4f  grad_thresh=%.4f  extent=%.4f",
            iteration, before, after, after - before,
            "stabilize" if iteration < 2000 else ("controlled" if iteration < 6000 else "normal"),
            min_opacity, grad_thresh, extent,
        )


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