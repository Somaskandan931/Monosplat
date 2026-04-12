"""
trainer.py
Training loop for 3D Gaussian Splatting.

Improvements:
- torch.autograd.set_detect_anomaly(True) enabled at training entry point.
- NaN loss check: skips iteration instead of crashing.
- Exponential LR decay applied every iteration.
- Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).
- Lower densification grad_threshold: 0.0001 (was 0.0002).
- densification_interval: 100 (driven from config).
- densify_until_iter extended to 70% of total iterations.
- densify_and_split called alongside densify_and_clone.
- Periodic render preview every 500 iterations saved to output directory.
- _reset_opacity and _densify_and_prune zero_grad and rebuild optimizer before
  any parameter replacement to cleanly discard autograd graph.
"""

import math
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .gaussian_model import GaussianModel
from .loss import combined_loss
from ..utils.io_utils import save_ply, save_checkpoint, load_checkpoint


class GaussianTrainer:
    """Trains a GaussianModel given training cameras and ground-truth images."""

    def __init__(
        self,
        model: GaussianModel,
        renderer,
        train_cameras,
        train_images,
        cfg,
    ):
        self.model    = model
        self.renderer = renderer
        self.cameras  = train_cameras
        self.images   = train_images
        self.cfg      = cfg

        self.output_dir = Path(cfg.training.output_dir)
        self.ckpt_dir   = Path(cfg.training.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Preview directory for periodic renders
        self.preview_dir = self.output_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.device == "cpu" and hasattr(cfg.training, 'iterations_cpu'):
            cfg.training.iterations = cfg.training.iterations_cpu
            print(f"[Trainer] CPU mode: iterations reduced to {cfg.training.iterations}")

        print(f"[Trainer] Device  : {self.device}")
        print(f"[Trainer] Output  : {self.output_dir.resolve()}")
        print(f"[Trainer] Ckpts   : {self.ckpt_dir.resolve()}")
        print(f"[Trainer] Previews: {self.preview_dir.resolve()}")

        self._grad_accum: Optional[torch.Tensor] = None
        self._grad_denom: Optional[torch.Tensor] = None
        self.last_metrics: dict = {}

        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Optimizer
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
        # Store base LRs for exponential decay
        self._base_lrs = {g["name"]: g["lr"] for g in param_groups}

    def _apply_lr_decay(self, iteration: int, total_iterations: int) -> None:
        """Exponential LR decay: lr = base_lr * exp(-5 * progress)."""
        progress = iteration / max(total_iterations, 1)
        decay    = math.exp(-5.0 * progress)
        for group in self.optimizer.param_groups:
            base = self._base_lrs.get(group["name"], group["lr"])
            group["lr"] = base * decay

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        start_iter: int = 0,
        on_iter_callback: Optional[Callable[[int, float], None]] = None,
        callback_every: int = 500,
    ) -> None:
        # Enable anomaly detection for debugging autograd issues
        torch.autograd.set_detect_anomaly(True)

        cfg        = self.cfg.training
        iterations = cfg.iterations
        save_every = cfg.save_every

        # Always derive from actual iterations — config value may be stale
        densify_until = int(iterations * 0.7)

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
            # Move to device on demand — keeps all training images on CPU to save VRAM
            gt_img = self.images[idx].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            rendered = self.renderer(self.model, camera)
            loss     = combined_loss(rendered, gt_img)

            # NaN check — skip iteration instead of crashing
            if torch.isnan(loss):
                nan_count += 1
                if nan_count <= 10 or nan_count % 100 == 0:
                    print(f"[Trainer] WARNING: NaN loss at iter {it} (total NaNs: {nan_count}). Skipping.")
                continue

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            if self.model._positions.grad is not None:
                with torch.no_grad():
                    grad_norm = self.model._positions.grad.norm(dim=1)
                    self._grad_accum += grad_norm
                    self._grad_denom += 1.0

            self.optimizer.step()

            # Exponential LR decay every iteration
            self._apply_lr_decay(it, iterations)

            loss_val      = loss.item()
            running_loss += loss_val

            if it % 1000 == 0:
                self.model.oneup_sh_degree()

            densify_from = getattr(cfg, "densify_from_iter", 500)
            densify_interval = getattr(cfg, "densification_interval", 100)

            if (densify_from <= it <= densify_until
                    and it % densify_interval == 0):
                self._densify_and_prune()

            opacity_reset_interval = getattr(cfg, "opacity_reset_interval", 2000)
            if it > 0 and it % opacity_reset_interval == 0 and it < densify_until:
                self._reset_opacity()

            if it > 0 and it % save_every == 0:
                self._save(it)

            # Periodic render preview every 500 iterations
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

            # Clear cache after preview and callbacks to reclaim any temporary allocations
            if self.device == "cuda" and it % 500 == 0:
                torch.cuda.empty_cache()

        self._save(iterations)
        print(f"[Trainer] Training complete. Model saved to {self.output_dir}")
        if nan_count > 0:
            print(f"[Trainer] Total NaN iterations skipped: {nan_count}")

        self.last_metrics = {
            "num_gaussians": len(self.model),
            "final_loss":    loss_val,
            "total_iters":   iterations,
            "nan_skipped":   nan_count,
        }

    # ------------------------------------------------------------------
    # Densification and pruning
    # ------------------------------------------------------------------

    def _densify_and_prune(self) -> None:
        avg_grads    = self._grad_accum / (self._grad_denom + 1e-8)
        scene_extent = self.model.positions.detach().norm(dim=1).max().item()

        # Drop autograd graph BEFORE touching parameters
        self.optimizer.zero_grad(set_to_none=True)

        # FIX: lower grad_threshold 0.0001 (was 0.0002) for more aggressive densification
        self.model.densify_and_clone(
            avg_grads, grad_threshold=0.0001, scene_extent=scene_extent
        )
        self.model.densify_and_split(
            avg_grads, grad_threshold=0.0001, scene_extent=scene_extent
        )

        prune_mask = (
            (self.model.opacities.squeeze() < 0.005) |
            (self.model.scales.max(dim=1).values > 0.1 * scene_extent)
        )

        max_gaussians = self.cfg.renderer.max_gaussians
        if len(self.model) > max_gaussians:
            n_prune    = len(self.model) - max_gaussians
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

        # Rebuild optimizer AFTER parameter replacement
        self._setup_optimizer()

        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Opacity reset
    # ------------------------------------------------------------------

    def _reset_opacity(self) -> None:
        """Reset opacities without any in-place mutation of a grad-tracked tensor."""
        # Step 1: drop current autograd graph
        self.optimizer.zero_grad(set_to_none=True)

        # Step 2: save reference to old parameter so we can clear its optimizer state
        old_opacity_param = self.model._opacities

        # Step 3: replace parameter fully out-of-place
        with torch.no_grad():
            new_opacities = old_opacity_param.detach().clamp(max=-4.595)
        self.model._opacities = torch.nn.Parameter(new_opacities)

        # Step 4: point the optimizer at the new Parameter and clear old state
        for group in self.optimizer.param_groups:
            if group.get("name") == "opacity":
                self.optimizer.state.pop(old_opacity_param, None)
                group["params"] = [self.model._opacities]
                break

        print("[Trainer] Opacity reset.")

    # ------------------------------------------------------------------
    # Preview render
    # ------------------------------------------------------------------

    def _save_preview(self, iteration: int, camera) -> None:
        """Save a preview render image every 500 iterations."""
        try:
            import numpy as np
            from PIL import Image
            with torch.no_grad():
                rendered = self.renderer(self.model, camera)
            img_np = rendered.detach().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            preview_path = self.preview_dir / f"preview_{iteration:06d}.png"
            Image.fromarray(img_np).save(str(preview_path))
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