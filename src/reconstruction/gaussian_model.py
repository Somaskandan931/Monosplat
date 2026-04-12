"""
gaussian_model.py
3D Gaussian Splatting model — Object / Product / Architecture mode.

Changes from MonoSplat v1 to match LeoDarcy/360GS API
------------------------------------------------------
- Added get_features() → returns full SH feature tensor (matches 360GS scene)
- Added get_covariance() → 3D covariance matrix (used by CUDA rasterizer)
- densify_grad_threshold now driven by config (360GS default: 0.0002)
- percent_dense controls clone/split threshold (360GS style)
- initial scale uses log(mean_dist) without *0.5 shrink for larger objects
- All .data usage replaced with .detach() to prevent autograd violations.
- prune_points / densify_and_clone always use .detach() before slicing/cat.
- densify_and_split added for high-grad large Gaussians.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1.0 - x))


def _knn_mean_dist(pts: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean distance to k nearest neighbours."""
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


def _build_rotation_matrix(rotations: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) → (N,3,3) rotation matrices."""
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
        torch.stack([2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
        torch.stack([2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], dim=1),
    ], dim=1)


class GaussianModel(nn.Module):
    """
    3D Gaussian Splatting scene representation — Object / Architecture mode.

    API matches LeoDarcy/360GS scene/gaussian_model.py so the same
    training loop and renderer calls work on both pipelines.

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

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.sh_degree        = sh_degree
        self.active_sh_degree = 0
        self.max_sh_degree    = sh_degree

        self._positions:     nn.Parameter = None
        self._features_dc:   nn.Parameter = None
        self._features_rest: nn.Parameter = None
        self._opacities:     nn.Parameter = None
        self._scales:        nn.Parameter = None
        self._rotations:     nn.Parameter = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create_from_points(self, positions: np.ndarray, colors: np.ndarray) -> None:
        n    = len(positions)
        pts  = torch.from_numpy(positions.astype(np.float32))
        cols = torch.from_numpy(colors.astype(np.float32))

        sh_dc         = (cols - 0.5) / self.SH_C0
        features_dc   = sh_dc.unsqueeze(1)
        n_rest        = (self.sh_degree + 1) ** 2 - 1
        features_rest = torch.zeros(n, n_rest, 3)

        # Object mode: use full mean_dist (no *0.5 shrink).
        # 360GS initialises with log(mean_dist) directly.
        mean_dist  = _knn_mean_dist(pts, k=3)
        log_scales = torch.log(mean_dist).unsqueeze(1).expand(n, 3).clone()

        opacities = inverse_sigmoid(torch.full((n, 1), 0.1))  # 360GS uses 0.1
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
    # 360GS-compatible properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> torch.Tensor:
        return self._positions

    # 360GS alias
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._positions

    @property
    def colors_sh(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._opacities)

    # 360GS alias
    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacities)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self._scales)

    # 360GS alias
    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scales)

    @property
    def rotations(self) -> torch.Tensor:
        return F.normalize(self._rotations, dim=1)

    # 360GS alias
    @property
    def get_rotation(self) -> torch.Tensor:
        return F.normalize(self._rotations, dim=1)

    def get_features(self) -> torch.Tensor:
        """Full SH feature tensor — matches 360GS get_features()."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """
        3D covariance matrices (N, 6) in upper-triangular packed form.
        Used by CUDA diff-gaussian-rasterization when available.
        Matches 360GS GaussianModel.get_covariance().
        """
        s   = self.scales * scaling_modifier          # (N, 3)
        rot = self.rotations                          # (N, 4)
        R   = _build_rotation_matrix(rot)             # (N, 3, 3)
        S   = s.unsqueeze(2) * torch.eye(3, device=s.device).unsqueeze(0)
        M   = R @ S                                   # (N, 3, 3)
        cov = M @ M.transpose(1, 2)                   # (N, 3, 3) symmetric
        # Pack upper triangle: (xx, xy, xz, yy, yz, zz)
        return torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
        ], dim=1)

    # ------------------------------------------------------------------
    # SH degree scheduling (matches 360GS)
    # ------------------------------------------------------------------

    def oneup_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"[GaussianModel] SH degree -> {self.active_sh_degree}")

    # ------------------------------------------------------------------
    # Density control (360GS-style with percent_dense)
    # ------------------------------------------------------------------

    def prune_points(self, mask: torch.Tensor) -> None:
        keep = ~mask
        self._positions     = nn.Parameter(self._positions.detach()[keep])
        self._features_dc   = nn.Parameter(self._features_dc.detach()[keep])
        self._features_rest = nn.Parameter(self._features_rest.detach()[keep])
        self._opacities     = nn.Parameter(self._opacities.detach()[keep])
        self._scales        = nn.Parameter(self._scales.detach()[keep])
        self._rotations     = nn.Parameter(self._rotations.detach()[keep])
        print(f"[GaussianModel] Pruned {mask.sum().item():,}. Remaining: {keep.sum().item():,}")

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        percent_dense: float = 0.01,
    ) -> None:
        """Clone under-reconstructed small Gaussians. Matches 360GS logic."""
        selected = (
            (grads >= grad_threshold) &
            (self.scales.detach().max(dim=1).values <= percent_dense * scene_extent)
        )
        n_clone = selected.sum().item()
        if n_clone == 0:
            return
        self._positions     = nn.Parameter(torch.cat([self._positions.detach(),     self._positions.detach()[selected]],     dim=0))
        self._features_dc   = nn.Parameter(torch.cat([self._features_dc.detach(),   self._features_dc.detach()[selected]],   dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.detach(), self._features_rest.detach()[selected]], dim=0))
        self._opacities     = nn.Parameter(torch.cat([self._opacities.detach(),     self._opacities.detach()[selected]],     dim=0))
        self._scales        = nn.Parameter(torch.cat([self._scales.detach(),        self._scales.detach()[selected]],        dim=0))
        self._rotations     = nn.Parameter(torch.cat([self._rotations.detach(),     self._rotations.detach()[selected]],     dim=0))
        print(f"[GaussianModel] Cloned {n_clone:,} Gaussians.")

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        percent_dense: float = 0.01,
        N: int = 2,
    ) -> None:
        """Split large over-reconstructed Gaussians. Matches 360GS split logic with N=2."""
        scales_det = self.scales.detach()
        selected = (
            (grads >= grad_threshold) &
            (scales_det.max(dim=1).values > percent_dense * scene_extent)
        )
        n_split = selected.sum().item()
        if n_split == 0:
            return

        stds  = scales_det[selected].repeat(N, 1)
        means = torch.zeros_like(stds)
        # Sample offsets along principal axes
        samples = torch.normal(mean=means, std=stds)

        rots   = _build_rotation_matrix(self._rotations.detach()[selected]).repeat(N, 1, 1)
        pos_new = (rots @ samples.unsqueeze(-1)).squeeze(-1) + self._positions.detach()[selected].repeat(N, 1)

        # Scale the children down (360GS divides by 0.8*N)
        new_log_scales = torch.log(scales_det[selected] / (0.8 * N)).repeat(N, 1)

        keep = ~selected
        self._positions     = nn.Parameter(torch.cat([self._positions.detach()[keep],     pos_new], dim=0))
        self._features_dc   = nn.Parameter(torch.cat([self._features_dc.detach()[keep],   self._features_dc.detach()[selected].repeat(N, 1, 1)], dim=0))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.detach()[keep], self._features_rest.detach()[selected].repeat(N, 1, 1)], dim=0))
        self._opacities     = nn.Parameter(torch.cat([self._opacities.detach()[keep],     self._opacities.detach()[selected].repeat(N, 1)], dim=0))
        self._scales        = nn.Parameter(torch.cat([self._scales.detach()[keep],        new_log_scales], dim=0))
        self._rotations     = nn.Parameter(torch.cat([self._rotations.detach()[keep],     self._rotations.detach()[selected].repeat(N, 1)], dim=0))
        print(f"[GaussianModel] Split {n_split:,} → {n_split*N:,} children.")

    # ------------------------------------------------------------------
    # Serialisation (unchanged)
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