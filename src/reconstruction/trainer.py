"""
trainer.py
Training loop for 3D Gaussian Splatting — Object / Product / Architecture mode.

Aligned with LeoDarcy/360GS training loop
-----------------------------------------
- percent_dense from config controls clone/split threshold (360GS style)
- densify_grad_threshold from config (360GS default: 0.0002)
- Exponential position LR decay using lambda scheduler (matches 360GS)
- densify_and_split uses N=2 (360GS default)
- Opacity reset every 3000 iters until densify_until_iter (360GS schedule)
- PSNR/SSIM evaluation on test cameras at eval_every intervals
- NaN loss check: skips iteration instead of crashing.
- Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).
"""

import math
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .gaussian_model import GaussianModel
from .loss import combined_loss, psnr_metric, ssim_metric
from ..utils.io_utils import save_ply, save_checkpoint, load_checkpoint


class GaussianTrainer:
    """Trains a GaussianModel — object-centric, 360GS-aligned."""

    def __init__(
        self,
        model: GaussianModel,
        renderer,
        train_cameras,
        train_images,
        cfg,
        test_cameras=None,
        test_images=None,
    ):
        self.model    = model
        self.renderer = renderer
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

        if self.device == "cpu" and hasattr(cfg.training, 'iterations_cpu'):
            cfg.training.iterations = cfg.training.iterations_cpu
            print(f"[Trainer] CPU mode: iterations → {cfg.training.iterations}")

        # Guard: if running on CPU without CUDA rasterizer, the software renderer
        # loop is O(N_gaussians) per pixel per iteration — impractical for large
        # iteration counts. Warn loudly so the user knows to switch to Colab/GPU.
        if self.device == "cpu":
            max_cpu_iters = getattr(cfg.training, 'iterations_cpu', 1000)
            if cfg.training.iterations > max_cpu_iters:
                import warnings
                warnings.warn(
                    f"[Trainer] CPU mode with {cfg.training.iterations:,} iterations and "
                    f"software renderer — this will take an impractical amount of time. "
                    f"Capping at {max_cpu_iters:,} (set training.iterations_cpu in config to change). "
                    "Use Colab/GPU for full training.",
                    RuntimeWarning, stacklevel=2,
                )
                cfg.training.iterations = max_cpu_iters

        print(f"[Trainer] Device  : {self.device}")
        print(f"[Trainer] Output  : {self.output_dir.resolve()}")

        self._grad_accum: Optional[torch.Tensor] = None
        self._grad_denom: Optional[torch.Tensor] = None
        self.last_metrics: dict = {}
        self.eval_log: list = []

        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Optimizer — matches 360GS param group structure
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
        """
        Exponential position LR decay matching 360GS:
            lr = base_lr * exp(-5 * progress)   [position only]
        Other param groups use fixed LR (360GS does not decay them).
        """
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
        torch.autograd.set_detect_anomaly(True)

        cfg        = self.cfg.training
        iterations = cfg.iterations
        save_every = cfg.save_every
        eval_every = getattr(cfg, "eval_every", 1000)

        densify_from     = getattr(cfg, "densify_from_iter", 500)
        densify_until    = getattr(cfg, "densify_until_iter", 15000)
        densify_interval = getattr(cfg, "densification_interval", 100)
        grad_threshold   = getattr(cfg, "densify_grad_threshold", 0.0002)  # 360GS default
        percent_dense    = getattr(cfg, "percent_dense", 0.01)             # 360GS default
        opacity_reset_interval = getattr(cfg, "opacity_reset_interval", 3000)
        lambda_dssim     = getattr(cfg, "lambda_dssim", 0.2)

        n = len(self.model)
        self._grad_accum = torch.zeros(n, device=self.device)
        self._grad_denom = torch.zeros(n, device=self.device)

        running_loss = 0.0
        loss_val     = 0.0
        nan_count    = 0
        pbar = tqdm(range(start_iter, iterations), desc="Training", dynamic_ncols=True)

        for it in pbar:
            idx    = np.random.randint(len(self.cameras))
            camera = self.cameras[idx]
            gt_img = self.images[idx].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            rendered = self.renderer(self.model, camera)
            loss     = combined_loss(rendered, gt_img, lambda_ssim=lambda_dssim)

            if torch.isnan(loss):
                nan_count += 1
                if nan_count <= 10 or nan_count % 100 == 0:
                    print(f"[Trainer] WARNING: NaN loss at iter {it} (total NaNs: {nan_count}). Skipping.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            if self.model._positions.grad is not None:
                with torch.no_grad():
                    grad_norm = self.model._positions.grad.norm(dim=1)
                    self._grad_accum += grad_norm
                    self._grad_denom += 1.0

            self.optimizer.step()
            self._apply_position_lr_decay(it, iterations)

            loss_val      = loss.item()
            running_loss += loss_val

            # ── SH degree scheduling (360GS: every 1000 iters) ──────
            if it % 1000 == 0:
                self.model.oneup_sh_degree()

            # ── Densification (360GS schedule) ──────────────────────
            if densify_from <= it <= densify_until and it % densify_interval == 0:
                self._densify_and_prune(
                    grad_threshold=grad_threshold,
                    percent_dense=percent_dense,
                )

            # ── Opacity reset (360GS schedule) ──────────────────────
            if it > 0 and it % opacity_reset_interval == 0 and it < densify_until:
                self._reset_opacity()

            # ── Checkpoint + PLY save ────────────────────────────────
            if it > 0 and it % save_every == 0:
                self._save(it)

            # ── Evaluation on test split (360GS style) ───────────────
            if it > 0 and it % eval_every == 0 and self.test_cameras:
                self._evaluate(it)

            # ── Preview render ───────────────────────────────────────
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
    # Densification + pruning (360GS-style)
    # ------------------------------------------------------------------

    def _densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        percent_dense: float  = 0.01,
    ) -> None:
        # Snapshot averaged gradients BEFORE any model size changes.
        # densify_and_clone and densify_and_split must both receive the
        # same avg_grads tensor (sized to the current model) — if clone
        # runs first it appends new Gaussians, making split receive a
        # stale tensor of the wrong length.
        avg_grads    = (self._grad_accum / (self._grad_denom + 1e-8)).clone()
        scene_extent = self.model.positions.detach().norm(dim=1).max().item()

        self.optimizer.zero_grad(set_to_none=True)

        self.model.densify_and_clone(
            avg_grads, grad_threshold=grad_threshold,
            scene_extent=scene_extent, percent_dense=percent_dense,
        )
        # Re-snapshot avg_grads for split: clone may have changed model size.
        # Newly cloned Gaussians have zero accumulated gradient so we pad.
        n_current = len(self.model)
        n_old     = avg_grads.shape[0]
        if n_current > n_old:
            padding   = torch.zeros(n_current - n_old, device=self.device)
            avg_grads_split = torch.cat([avg_grads, padding], dim=0)
        else:
            avg_grads_split = avg_grads

        self.model.densify_and_split(
            avg_grads_split, grad_threshold=grad_threshold,
            scene_extent=scene_extent, percent_dense=percent_dense,
            N=2,   # 360GS uses N=2
        )

        # Prune: opacity too low OR Gaussian too large
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
        self._grad_accum = torch.zeros(n, device=self.device)
        self._grad_denom = torch.zeros(n, device=self.device)
        self._setup_optimizer()

        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Opacity reset (360GS style)
    # ------------------------------------------------------------------

    def _reset_opacity(self) -> None:
        """
        Reset opacities to a small value following the 360GS schedule.
        sigmoid(-4.595) ≈ 0.01 — the reset ceiling, NOT a floor.
        We cap each logit at -4.595 so well-trained high-opacity Gaussians
        are brought back to ~0.01 without pushing low-opacity ones even lower.
        """
        self.optimizer.zero_grad(set_to_none=True)
        old_opacity_param = self.model._opacities
        reset_logit = torch.tensor(-4.595, device=self.device, dtype=old_opacity_param.dtype)
        with torch.no_grad():
            # torch.min: caps each logit at reset_logit (≈ opacity 0.01)
            # clamp(max=) would force ALL opacities to ≤ 0.01, destroying training.
            new_opacities = torch.min(old_opacity_param.detach(), reset_logit.expand_as(old_opacity_param))
        self.model._opacities = torch.nn.Parameter(new_opacities)
        for group in self.optimizer.param_groups:
            if group.get("name") == "opacity":
                self.optimizer.state.pop(old_opacity_param, None)
                group["params"] = [self.model._opacities]
                break
        print("[Trainer] Opacity reset.")

    # ------------------------------------------------------------------
    # Evaluation (PSNR + SSIM — matches 360GS metrics.py)
    # ------------------------------------------------------------------

    def _evaluate(self, iteration: int) -> None:
        self.model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for cam, gt in zip(self.test_cameras, self.test_images):
                rendered = self.renderer(self.model, cam)
                gt_dev   = gt.to(self.device)
                psnrs.append(psnr_metric(rendered, gt_dev).item())
                ssims.append((1.0 - ssim_metric(rendered, gt_dev)).item())  # ssim_metric returns (1-SSIM)
        self.model.train()
        avg_psnr = np.mean(psnrs)
        avg_ssim = 1.0 - np.mean(ssims)  # convert back to SSIM
        record = {"iter": iteration, "psnr": round(avg_psnr, 3), "ssim": round(avg_ssim, 4)}
        self.eval_log.append(record)
        print(f"[Eval  ] iter={iteration:6d}  PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}")

    # ------------------------------------------------------------------
    # Preview render
    # ------------------------------------------------------------------

    def _save_preview(self, iteration: int, camera) -> None:
        try:
            import numpy as np
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