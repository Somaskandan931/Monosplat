"""
gaussian_model.py
Defines the 3D Gaussian Splatting model as a PyTorch nn.Module.

Improvements:
- All .data usage replaced with .detach() to prevent autograd violations.
- No tensor modified in-place if it participates in gradients.
- prune_points / densify_and_clone always use .detach() before slicing/cat,
  then wrap results in nn.Parameter.
- densify_and_split added for high-grad large Gaussians.
- Reduced initial scale (x0.5) and opacity (0.05) for better initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1.0 - x))


def _knn_mean_dist(pts: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean distance to k nearest neighbours. Robust for any N >= 1."""
    N = pts.shape[0]
    if N == 0:
        return torch.tensor([], device=pts.device)
    k_actual = min(k, N - 1)
    if k_actual == 0:
        return torch.zeros(N, device=pts.device)
    dist_matrix = torch.cdist(pts, pts)
    dist_matrix.fill_diagonal_(float('inf'))
    knn_dists = torch.topk(dist_matrix, k=k_actual, largest=False, dim=1).values
    return knn_dists.mean(dim=1).clamp(min=1e-7)


class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting scene representation.

    Learnable parameters
    --------------------
    _positions      (N, 3)
    _features_dc    (N, 1, 3)
    _features_rest  (N, (deg+1)^2-1, 3)
    _opacities      (N, 1)
    _scales         (N, 3)
    _rotations      (N, 4)
    """

    SH_C0 = 0.28209479177387814

    def __init__(self, sh_degree: int = 1):
        super().__init__()
        self.sh_degree        = sh_degree
        self.active_sh_degree = 0

        self._positions:     nn.Parameter | None = None
        self._features_dc:   nn.Parameter | None = None
        self._features_rest: nn.Parameter | None = None
        self._opacities:     nn.Parameter | None = None
        self._scales:        nn.Parameter | None = None
        self._rotations:     nn.Parameter | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create_from_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        n    = len(positions)
        pts  = torch.from_numpy(positions.astype(np.float32))
        cols = torch.from_numpy(colors.astype(np.float32))

        sh_dc         = (cols - 0.5) / self.SH_C0
        features_dc   = sh_dc.unsqueeze(1)
        features_rest = torch.zeros(n, (self.sh_degree + 1) ** 2 - 1, 3)

        # FIX: scale * 0.5 for tighter initial Gaussians
        mean_dist  = _knn_mean_dist(pts, k=3)
        log_scales = torch.log(mean_dist * 0.5).unsqueeze(1).expand(n, 3).clone()

        # Low initial opacity — avoids floaters early in training
        opacities = inverse_sigmoid(torch.full((n, 1), 0.05))
        rotations = torch.zeros(n, 4)
        rotations[:, 0] = 1.0

        self._positions     = nn.Parameter(pts)
        self._features_dc   = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._opacities     = nn.Parameter(opacities)
        self._scales        = nn.Parameter(log_scales)
        self._rotations     = nn.Parameter(rotations)

        print(f"[GaussianModel] Initialised {n:,} Gaussians from point cloud.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> torch.Tensor:
        return self._positions

    @property
    def colors_sh(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._opacities)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._scales)

    @property
    def rotations(self) -> torch.Tensor:
        return F.normalize(self._rotations, dim=1)

    # ------------------------------------------------------------------
    # SH degree scheduling
    # ------------------------------------------------------------------

    def oneup_sh_degree(self) -> None:
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1
            print(f"[GaussianModel] SH degree -> {self.active_sh_degree}")

    # ------------------------------------------------------------------
    # Density control
    # ------------------------------------------------------------------

    def prune_points(self, mask: torch.Tensor) -> None:
        """Remove Gaussians where mask is True. All ops use .detach()."""
        keep = ~mask
        self._positions     = nn.Parameter(self._positions.detach()[keep])
        self._features_dc   = nn.Parameter(self._features_dc.detach()[keep])
        self._features_rest = nn.Parameter(self._features_rest.detach()[keep])
        self._opacities     = nn.Parameter(self._opacities.detach()[keep])
        self._scales        = nn.Parameter(self._scales.detach()[keep])
        self._rotations     = nn.Parameter(self._rotations.detach()[keep])
        print(f"[GaussianModel] Pruned {mask.sum().item():,} Gaussians. "
              f"Remaining: {keep.sum().item():,}")

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
    ) -> None:
        """Clone small under-reconstructed Gaussians. All ops use .detach()."""
        selected = (
            (grads >= grad_threshold) &
            (self.scales.detach().max(dim=1).values <= 0.01 * scene_extent)
        )
        n_clone = selected.sum().item()
        if n_clone == 0:
            return

        self._positions     = nn.Parameter(torch.cat([
            self._positions.detach(),
            self._positions.detach()[selected]], dim=0))
        self._features_dc   = nn.Parameter(torch.cat([
            self._features_dc.detach(),
            self._features_dc.detach()[selected]], dim=0))
        self._features_rest = nn.Parameter(torch.cat([
            self._features_rest.detach(),
            self._features_rest.detach()[selected]], dim=0))
        self._opacities     = nn.Parameter(torch.cat([
            self._opacities.detach(),
            self._opacities.detach()[selected]], dim=0))
        self._scales        = nn.Parameter(torch.cat([
            self._scales.detach(),
            self._scales.detach()[selected]], dim=0))
        self._rotations     = nn.Parameter(torch.cat([
            self._rotations.detach(),
            self._rotations.detach()[selected]], dim=0))
        print(f"[GaussianModel] Cloned {n_clone:,} Gaussians.")

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
    ) -> None:
        """Split large high-grad Gaussians into two smaller children."""
        scales_det = self.scales.detach()
        selected = (
            (grads >= grad_threshold) &
            (scales_det.max(dim=1).values > 0.01 * scene_extent)
        )
        n_split = selected.sum().item()
        if n_split == 0:
            return

        stds  = scales_det[selected]
        rots  = self._rotations.detach()[selected]
        means = self._positions.detach()[selected]

        w, x, y, z = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
            torch.stack([2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
            torch.stack([2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], dim=1),
        ], dim=1)

        local_offset = torch.zeros_like(stds)
        local_offset[:, 0] = stds[:, 0]
        world_offset = (R @ local_offset.unsqueeze(2)).squeeze(2)

        pos_a = means + world_offset
        pos_b = means - world_offset
        new_scales = torch.log((stds / 1.6).clamp(min=1e-7))

        keep = ~selected
        self._positions     = nn.Parameter(torch.cat([
            self._positions.detach()[keep], pos_a, pos_b], dim=0))
        self._features_dc   = nn.Parameter(torch.cat([
            self._features_dc.detach()[keep],
            self._features_dc.detach()[selected],
            self._features_dc.detach()[selected]], dim=0))
        self._features_rest = nn.Parameter(torch.cat([
            self._features_rest.detach()[keep],
            self._features_rest.detach()[selected],
            self._features_rest.detach()[selected]], dim=0))
        self._opacities     = nn.Parameter(torch.cat([
            self._opacities.detach()[keep],
            self._opacities.detach()[selected],
            self._opacities.detach()[selected]], dim=0))
        self._scales        = nn.Parameter(torch.cat([
            self._scales.detach()[keep], new_scales, new_scales], dim=0))
        self._rotations     = nn.Parameter(torch.cat([
            self._rotations.detach()[keep],
            self._rotations.detach()[selected],
            self._rotations.detach()[selected]], dim=0))
        print(f"[GaussianModel] Split {n_split:,} Gaussians -> {n_split*2:,} children.")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "positions":  self._positions.detach().cpu().numpy(),
            "colors":     (self._features_dc[:, 0, :] * self.SH_C0 + 0.5).detach().cpu().numpy(),
            "opacities":  torch.sigmoid(self._opacities).detach().cpu().numpy(),
            "scales":     torch.exp(self._scales).detach().cpu().numpy(),
            "rotations":  F.normalize(self._rotations, dim=1).detach().cpu().numpy(),
        }

    def load_state(self, state: Dict[str, np.ndarray]) -> None:
        self.create_from_points(state["positions"], state["colors"])
        with torch.no_grad():
            # .detach() inside no_grad — safe, no autograd participation
            self._opacities.detach().copy_(
                inverse_sigmoid(torch.from_numpy(state["opacities"]).clamp(1e-6, 1 - 1e-6))
            )
            self._scales.detach().copy_(
                torch.log(torch.from_numpy(state["scales"]).clamp(min=1e-7))
            )
            self._rotations.detach().copy_(torch.from_numpy(state["rotations"]))

    def __len__(self) -> int:
        if self._positions is None:
            return 0
        return self._positions.shape[0]