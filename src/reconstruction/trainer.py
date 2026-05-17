"""
trainer.py
Training loop for 3D Gaussian Splatting — compatible with gsplat backend.

Key differences from the original diff-gaussian-rasterization approach
----------------------------------------------------------------------
- When gsplat is available, we use gsplat.rasterization() directly in the
  training loop so we can access `meta["radii"]` and `meta["means2d"]` for
  gradient-based densification — the same information 3DGS extracts from
  screenspace_points.grad in the CUDA extension.
- When gsplat is not available we fall back to the software renderer, but
  densification is estimated from position gradients (less accurate but
  still functional for CPU-only runs / debugging).
- PSNR/SSIM evaluation on held-out test cameras every eval_every iters.
- NaN loss detection with skip and counter.
- Exponential position LR decay matching the original 3DGS paper.
"""

import math
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .gaussian_model import GaussianModel
from .loss import combined_loss, psnr_metric, ssim_metric
from ..utils.io_utils import save_ply, save_checkpoint, load_checkpoint


def _try_import_gsplat():
    try:
        import gsplat
        return gsplat
    except ImportError:
        return None


_GSPLAT = _try_import_gsplat()


def _eval_sh_colors(model, positions, camera_position, device):
    """Evaluate SH colours for current view direction."""
    from ..renderer.renderer import _eval_sh, SH_C0
    sh_coeffs = model.get_features().to(device)
    # Ensure sh_coeffs has shape (N, K, 3) - handle case where it's (N, 3)
    if sh_coeffs.dim() == 2:
        sh_coeffs = sh_coeffs.unsqueeze(1)
    cam_pos   = torch.from_numpy(camera_position).to(device, dtype=torch.float32)
    view_dirs = F.normalize(positions - cam_pos.unsqueeze(0), dim=1)
    return _eval_sh(model.active_sh_degree, sh_coeffs, view_dirs)


class GaussianTrainer:
    """Trains a GaussianModel using gsplat when available."""

    def __init__(
        self,
        model: GaussianModel,
        renderer,            # GaussianRenderer instance or callable
        train_cameras,
        train_images,
        cfg,
        test_cameras=None,
        test_images=None,
    ):
        self.model    = model
        self.renderer = renderer   # used for software-path and eval renders
        self.cameras  = train_cameras
        self.images   = train_images
        self.test_cameras = test_cameras or []
        self.test_images  = test_images  or []
        self.cfg      = cfg

        self.output_dir = Path(cfg.training.output_dir)
        self.ckpt_dir   = Path(cfg.training.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.preview_dir = self.output_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # CPU guard
        if self.device == "cpu":
            max_cpu_iters = getattr(cfg.training, 'iterations_cpu', 1000)
            if cfg.training.iterations > max_cpu_iters:
                import warnings
                warnings.warn(
                    f"[Trainer] CPU mode — capping iterations at {max_cpu_iters:,}. "
                    "Use Colab/GPU for full training.",
                    RuntimeWarning, stacklevel=2,
                )
                cfg.training.iterations = max_cpu_iters

        # Decide whether to use the gsplat training path
        self._use_gsplat_train = (
            _GSPLAT is not None and self.device == "cuda"
        )

        print(f"[Trainer] Device  : {self.device}")
        print(f"[Trainer] Backend : {'gsplat' if self._use_gsplat_train else 'software renderer'}")
        print(f"[Trainer] Output  : {self.output_dir.resolve()}")

        self._grad_accum: Optional[torch.Tensor] = None
        self._grad_denom: Optional[torch.Tensor] = None
        self._radii_accum: Optional[torch.Tensor] = None   # gsplat path
        self.last_metrics: dict = {}
        self.eval_log: list = []

        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Optimizer — Adam with per-parameter groups
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        lr = self.cfg.training.learning_rate
        param_groups = [
            {"params": [self.model._positions],     "lr": lr.position,       "name": "position"},
            {"params": [self.model._features_dc],   "lr": lr.feature,        "name": "feature_dc"},
            {"params": [self.model._features_rest], "lr": lr.feature / 20.0, "name": "feature_rest"},
            {"params": [self.model._opacities],     "lr": lr.opacity,        "name": "opacity"},
            {"params": [self.model._scales],        "lr": lr.scaling,        "name": "scaling"},
            {"params": [self.model._rotations],     "lr": lr.rotation,       "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self._base_lrs = {g["name"]: g["lr"] for g in param_groups}

    def _apply_position_lr_decay(self, iteration: int, total_iterations: int) -> None:
        """Exponential position LR decay: lr = base_lr * exp(-5 * progress)."""
        progress = iteration / max(total_iterations, 1)
        decay    = math.exp(-5.0 * progress)
        for group in self.optimizer.param_groups:
            if group["name"] == "position":
                group["lr"] = self._base_lrs["position"] * decay

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        start_iter: int = 0,
        on_iter_callback: Optional[Callable[[int, float], None]] = None,
        callback_every: int = 500,
    ) -> None:
        torch.autograd.set_detect_anomaly(False)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        cfg        = self.cfg.training
        iterations = cfg.iterations
        save_every = cfg.save_every
        eval_every = getattr(cfg, "eval_every", 1000)

        densify_from     = getattr(cfg, "densify_from_iter", 500)
        densify_until    = getattr(cfg, "densify_until_iter", 15000)
        densify_interval = getattr(cfg, "densification_interval", 100)
        grad_threshold   = getattr(cfg, "densify_grad_threshold", 0.0002)
        percent_dense    = getattr(cfg, "percent_dense", 0.01)
        opacity_reset_interval = getattr(cfg, "opacity_reset_interval", 3000)
        lambda_dssim     = getattr(cfg, "lambda_dssim", 0.2)

        n = len(self.model)
        self._grad_accum  = torch.zeros(n, device=self.device)
        self._grad_denom  = torch.zeros(n, device=self.device)
        self._radii_accum = torch.zeros(n, device=self.device)

        running_loss = 0.0
        loss_val     = 0.0
        nan_count    = 0
        pbar = tqdm(range(start_iter, iterations), desc="Training", dynamic_ncols=True)

        for it in pbar:
            idx    = np.random.randint(len(self.cameras))
            camera = self.cameras[idx]
            gt_img = self.images[idx].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self._use_gsplat_train:
                rendered, meta = self._gsplat_forward(camera)
            else:
                rendered = self.renderer(self.model, camera)
                meta     = None

            loss = combined_loss(rendered, gt_img, lambda_ssim=lambda_dssim)

            if torch.isnan(loss):
                nan_count += 1
                if nan_count <= 10 or nan_count % 100 == 0:
                    print(f"[Trainer] NaN loss at iter {it} (total: {nan_count}). Skipping.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Accumulate gradients for densification
            with torch.no_grad():
                if meta is not None and "means2d" in meta:
                    # gsplat path: use 2D mean gradients (more accurate)
                    means2d = meta["means2d"]
                    if means2d.grad is not None:
                        grad_norm = means2d.grad.norm(dim=-1)
                        # means2d may be packed — scatter to full N
                        if "gaussian_ids" in meta:
                            gids = meta["gaussian_ids"]
                            self._grad_accum.scatter_add_(0, gids, grad_norm)
                            self._grad_denom.scatter_add_(0, gids, torch.ones_like(grad_norm))
                        else:
                            n_cur = min(grad_norm.shape[0], len(self._grad_accum))
                            self._grad_accum[:n_cur] += grad_norm[:n_cur]
                            self._grad_denom[:n_cur] += 1.0
                elif self.model._positions.grad is not None:
                    # Software path: use 3D position gradients
                    grad_norm = self.model._positions.grad.norm(dim=1)
                    self._grad_accum += grad_norm
                    self._grad_denom += 1.0

                # Track radii for visibility-based pruning
                if meta is not None and "radii" in meta:
                    radii = meta["radii"]
                    if "gaussian_ids" in meta:
                        self._radii_accum.scatter_add_(0, meta["gaussian_ids"],
                                                        radii.float())
                    else:
                        n_cur = min(radii.shape[0], len(self._radii_accum))
                        self._radii_accum[:n_cur] = torch.max(
                            self._radii_accum[:n_cur], radii[:n_cur].float()
                        )

            self.optimizer.step()
            self._apply_position_lr_decay(it, iterations)

            loss_val      = loss.item()
            running_loss += loss_val

            # SH degree scheduling
            if it % 1000 == 0:
                self.model.oneup_sh_degree()

            # Densification
            if densify_from <= it <= densify_until and it % densify_interval == 0:
                self._densify_and_prune(
                    grad_threshold=grad_threshold,
                    percent_dense=percent_dense,
                )

            # Opacity reset
            if it > 0 and it % opacity_reset_interval == 0 and it < densify_until:
                self._reset_opacity()

            # Checkpoint
            if it > 0 and it % save_every == 0:
                self._save(it)

            # Eval
            if it > 0 and it % eval_every == 0 and self.test_cameras:
                self._evaluate(it)

            # Preview
            if it > 0 and it % 500 == 0:
                self._save_preview(it, camera)

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "N": f"{len(self.model):,}", "NaN": nan_count})

            if on_iter_callback and it > 0 and it % callback_every == 0:
                avg_loss     = running_loss / callback_every
                running_loss = 0.0
                try:
                    on_iter_callback(it, avg_loss)
                except Exception:
                    pass

            if self.device == "cuda" and it % 500 == 0:
                torch.cuda.empty_cache()

        self._save(iterations)
        print(f"[Trainer] Training complete. Output: {self.output_dir}")
        if nan_count:
            print(f"[Trainer] Total NaN iterations skipped: {nan_count}")

        self.last_metrics = {
            "num_gaussians": len(self.model),
            "final_loss":    loss_val,
            "total_iters":   iterations,
            "nan_skipped":   nan_count,
        }

    # ------------------------------------------------------------------
    # gsplat forward pass (returns image + meta for densification)
    # ------------------------------------------------------------------

    def _gsplat_forward(self, camera) -> tuple:
        """
        Run a gsplat rasterization forward pass.
        Returns (rendered_image [3,H,W], meta dict).
        The meta dict contains 'means2d', 'radii', 'gaussian_ids' for densification.
        """
        gs = _GSPLAT

        positions = self.model.positions.to(self.device)
        quats     = self.model.get_rotation.to(self.device)
        scales    = self.model.get_scaling.to(self.device)
        opacities = self.model.get_opacity.to(self.device).squeeze(-1)

        # Evaluate colours for this view
        cam_pos   = torch.from_numpy(camera.position).to(self.device, dtype=torch.float32)
        view_dirs = F.normalize(positions - cam_pos.unsqueeze(0), dim=1)
        from ..renderer.renderer import _eval_sh
        sh_coeffs = self.model.get_features().to(self.device)
        # Ensure sh_coeffs has shape (N, K, 3) - handle case where it's (N, 3)
        if sh_coeffs.dim() == 2:
            sh_coeffs = sh_coeffs.unsqueeze(1)
        colors    = _eval_sh(self.model.active_sh_degree, sh_coeffs, view_dirs)

        fx = torch.tensor(camera.fx, device=self.device, dtype=torch.float32)
        fy = torch.tensor(camera.fy, device=self.device, dtype=torch.float32)
        cx = torch.tensor(camera.cx, device=self.device, dtype=torch.float32)
        cy = torch.tensor(camera.cy, device=self.device, dtype=torch.float32)

        viewmat = torch.from_numpy(camera.world_view_transform).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)
        Ks = torch.zeros(1, 3, 3, device=self.device, dtype=torch.float32)
        Ks[0, 0, 0] = fx; Ks[0, 1, 1] = fy
        Ks[0, 0, 2] = cx; Ks[0, 1, 2] = cy; Ks[0, 2, 2] = 1.0

        bg = self.renderer.bg_color.unsqueeze(0) if hasattr(self.renderer, 'bg_color') else \
             torch.ones(1, 3, device=self.device)

        render_colors, render_alphas, meta = gs.rasterization(
            means    = positions,
            quats    = quats,
            scales   = scales,
            opacities= opacities,
            colors   = colors,
            viewmats = viewmat,
            Ks       = Ks,
            width    = camera.image_width,
            height   = camera.image_height,
            sh_degree= 0,          # already evaluated above
            near_plane  = camera.near,
            far_plane   = camera.far,
            backgrounds = bg,
            packed   = True,
            render_mode = "RGB",
        )

        rendered = render_colors[0].permute(2, 0, 1).clamp(0, 1)
        return rendered, meta

    # ------------------------------------------------------------------
    # Densification + pruning
    # ------------------------------------------------------------------

    def _densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        percent_dense: float  = 0.01,
    ) -> None:
        avg_grads    = (self._grad_accum / (self._grad_denom + 1e-8)).clone()
        scene_extent = self.model.positions.detach().norm(dim=1).max().item()

        self.optimizer.zero_grad(set_to_none=True)

        self.model.densify_and_clone(
            avg_grads, grad_threshold=grad_threshold,
            scene_extent=scene_extent, percent_dense=percent_dense,
        )

        n_current = len(self.model)
        n_old     = avg_grads.shape[0]
        if n_current > n_old:
            padding         = torch.zeros(n_current - n_old, device=self.device)
            avg_grads_split = torch.cat([avg_grads, padding], dim=0)
        else:
            avg_grads_split = avg_grads

        self.model.densify_and_split(
            avg_grads_split, grad_threshold=grad_threshold,
            scene_extent=scene_extent, percent_dense=percent_dense,
            N=2,
        )

        # Prune: low opacity OR too large
        prune_mask = (
            (self.model.opacities.squeeze() < 0.005) |
            (self.model.scales.max(dim=1).values > 0.1 * scene_extent)
        )

        max_gaussians = self.cfg.renderer.max_gaussians
        if len(self.model) > max_gaussians:
            n_prune = len(self.model) - max_gaussians
            opacity_flat = self.model.opacities.squeeze()
            _, low_idx = opacity_flat.topk(n_prune, largest=False)
            budget_mask = torch.zeros(len(self.model), dtype=torch.bool, device=self.device)
            budget_mask[low_idx] = True
            prune_mask = prune_mask | budget_mask

        if prune_mask.any():
            self.model.prune_points(prune_mask)

        n = len(self.model)
        self._grad_accum  = torch.zeros(n, device=self.device)
        self._grad_denom  = torch.zeros(n, device=self.device)
        self._radii_accum = torch.zeros(n, device=self.device)
        self._setup_optimizer()

        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Opacity reset
    # ------------------------------------------------------------------

    def _reset_opacity(self) -> None:
        """Reset opacities to ~0.01 (sigmoid(-4.595) ≈ 0.01)."""
        self.optimizer.zero_grad(set_to_none=True)
        old_opacity_param = self.model._opacities
        reset_logit = torch.tensor(-4.595, device=self.device, dtype=old_opacity_param.dtype)
        with torch.no_grad():
            new_opacities = torch.min(
                old_opacity_param.detach(),
                reset_logit.expand_as(old_opacity_param)
            )
        self.model._opacities = torch.nn.Parameter(new_opacities)
        for group in self.optimizer.param_groups:
            if group.get("name") == "opacity":
                self.optimizer.state.pop(old_opacity_param, None)
                group["params"] = [self.model._opacities]
                break
        print("[Trainer] Opacity reset.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, iteration: int) -> None:
        self.model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for cam, gt in zip(self.test_cameras, self.test_images):
                rendered = self.renderer(self.model, cam)
                gt_dev   = gt.to(self.device)
                psnrs.append(psnr_metric(rendered, gt_dev).item())
                ssims.append(1.0 - ssim_metric(rendered, gt_dev).item())
        self.model.train()
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        record = {"iter": iteration, "psnr": round(avg_psnr, 3), "ssim": round(avg_ssim, 4)}
        self.eval_log.append(record)
        print(f"[Eval  ] iter={iteration:6d}  PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}")

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _save_preview(self, iteration: int, camera) -> None:
        try:
            from PIL import Image
            with torch.no_grad():
                rendered = self.renderer(self.model, camera)
            img_np = rendered.detach().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(str(self.preview_dir / f"preview_{iteration:06d}.png"))
        except Exception as e:
            print(f"[Trainer] Preview save failed at iter {iteration}: {e}")

    # ------------------------------------------------------------------
    # Save / resume
    # ------------------------------------------------------------------

    def _save(self, iteration: int) -> None:
        ply_path  = self.output_dir / f"point_cloud_iter_{iteration:06d}.ply"
        ckpt_path = self.ckpt_dir   / f"checkpoint_{iteration:06d}.pkl"
        save_ply(str(ply_path), self.model.get_state())
        save_checkpoint(str(ckpt_path), {
            "iteration":   iteration,
            "model_state": self.model.state_dict(),
        })

    def resume_from_checkpoint(self, ckpt_path: str) -> int:
        state = load_checkpoint(ckpt_path)
        self.model.load_state_dict(state["model_state"])
        print(f"[Trainer] Resumed from iteration {state['iteration']}")
        return state["iteration"]