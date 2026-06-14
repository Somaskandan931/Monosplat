"""
gaussian_model.py — 3D Gaussian representation.

Based on the original graphdeco-inria/gaussian-splatting implementation
(https://github.com/graphdeco-inria/gaussian-splatting), adapted for
MonoSplat and Tanks-and-Temples style footage.

Key design principles (from original paper):
  - Log-space scale parameterisation prevents negative scales
  - Sigmoid opacity prevents out-of-range values
  - Quaternion rotation with normalisation
  - SH-DC stored as (rgb - 0.5) / SH_C0 so _eval_sh round-trips correctly
  - Adam optimizer state patching for densification (clone / split / prune)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

SH_C0 = 0.28209479177387814   # zeroth-order SH coefficient


class GaussianModel(nn.Module):
    """Learnable set of 3-D Gaussians for differentiable splatting."""

    def __init__(self, sh_degree: int = 3) -> None:
        super().__init__()
        self.sh_degree        = sh_degree
        self.active_sh_degree = 0
        self.max_sh_degree    = sh_degree

        # Learnable parameters — populated by initialise_from_pcd or checkpoint
        self._xyz:           Tensor = torch.empty(0)
        self._features_dc:   Tensor = torch.empty(0)
        self._features_rest: Tensor = torch.empty(0)
        self._scales:        Tensor = torch.empty(0)   # log-space
        self._rotations:     Tensor = torch.empty(0)   # quaternion (unnormalised)
        self._opacities:     Tensor = torch.empty(0)   # logit-space

        # Densification accumulators — set in initialise_from_pcd
        self.xyz_gradient_accum: Tensor = torch.empty(0)
        self.denom:              Tensor = torch.empty(0)
        self.max_radii2D:        Tensor = torch.empty(0)

        # Scale ceiling derived from scene extent (set in initialise_from_pcd)
        self._max_log_scale: float = 0.0

    # ------------------------------------------------------------------
    # Activated properties used during rendering
    # ------------------------------------------------------------------

    @property
    def get_xyz(self) -> Tensor:
        return self._xyz

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def get_features(self) -> Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_scaling(self) -> Tensor:
        return torch.exp(
            torch.clamp(self._scales, min=-7.0, max=self._max_log_scale)
        )

    @property
    def get_rotation(self) -> Tensor:
        return F.normalize(self._rotations, dim=-1, eps=1e-8)

    @property
    def get_opacity(self) -> Tensor:
        return torch.sigmoid(self._opacities)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._xyz.shape[0]

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = self._xyz.device
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
        self.denom               = self.denom.to(device)
        self.max_radii2D         = self.max_radii2D.to(device)
        return module

    def one_up_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_state(self) -> dict:
        """Return activated Gaussian tensors as CPU numpy arrays for export."""
        with torch.no_grad():
            colors = (self._features_dc[:, 0, :] * SH_C0 + 0.5).clamp(0.0, 1.0)
            return {
                "positions": self.get_xyz.detach().cpu().numpy(),
                "sh_dc":     self._features_dc.detach().cpu().numpy(),
                "sh_rest":   self._features_rest.detach().cpu().numpy(),
                "colors":    colors.detach().cpu().numpy(),
                "opacities": self.get_opacity.detach().cpu().numpy(),
                "scales":    self.get_scaling.detach().cpu().numpy(),
                "rotations": self.get_rotation.detach().cpu().numpy(),
            }

    # ------------------------------------------------------------------
    # Initialisation from a point cloud
    # ------------------------------------------------------------------

    def initialise_from_pcd(
        self,
        xyz: Tensor,
        rgb: Tensor,
        spatial_lr_scale: float = 1.0,
    ) -> None:
        """
        Populate model parameters from a coloured point cloud.

        Args:
            xyz:              (N, 3) point positions in normalised scene space.
            rgb:              (N, 3) colours in [0, 1].
            spatial_lr_scale: cameras_extent after normalisation (floor 1.0).
                              Gaussians start at ~10% of scene radius.
        """
        self.spatial_lr_scale = spatial_lr_scale

        # Scale ceiling: each Gaussian starts at ≤ 10% of scene radius
        self._max_log_scale = math.log(max(spatial_lr_scale * 0.1, 1e-4))

        n      = xyz.shape[0]
        device = xyz.device

        # SH-DC: convert RGB [0,1] → (rgb - 0.5) / SH_C0 so _eval_sh round-trips
        features_dc = (rgb.unsqueeze(1) - 0.5) / SH_C0  # (N, 1, 3)

        # Higher-order SH bands initialised to zero
        n_rest = (self.sh_degree + 1) ** 2 - 1
        features_rest = torch.zeros(n, n_rest, 3, dtype=torch.float32, device=device)

        # Nearest-neighbour distances → initial Gaussian scales
        dist2 = _distCUDA2(xyz.float())
        dist2 = dist2.clamp(min=1e-7)
        scales_init = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales_init = scales_init.clamp(
            min=-7.0, max=self._max_log_scale
        )

        # Unit quaternions (no rotation)
        rots = torch.zeros(n, 4, dtype=torch.float32, device=device)
        rots[:, 0] = 1.0

        # Initial opacity: logit(0.1) ≈ -2.197
        opacities = _inverse_sigmoid(
            torch.full((n, 1), 0.1, dtype=torch.float32, device=device)
        )

        self._xyz           = nn.Parameter(xyz.float())
        self._features_dc   = nn.Parameter(features_dc.float())
        self._features_rest = nn.Parameter(features_rest.float())
        self._scales        = nn.Parameter(scales_init.float())
        self._rotations     = nn.Parameter(rots.float())
        self._opacities     = nn.Parameter(opacities.float())

        # Densification accumulators
        self.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        self.denom               = torch.zeros(n, 1, device=device)
        self.max_radii2D         = torch.zeros(n,    device=device)

    # ------------------------------------------------------------------
    # Opacity reset (called by Trainer every opacity_reset_interval iters)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Gradient accumulation
    # ------------------------------------------------------------------

    def update_stats(self, radii: Tensor, viewspace_grad: Tensor) -> None:
        """Legacy API: dense (N,) radii and (N, 2) viewspace gradients."""
        self.update_stats_norm(radii, viewspace_grad.norm(dim=-1))

    def update_stats_norm(self, radii: Tensor, grad_norm: Tensor) -> None:
        """
        Accumulate pre-normed gradient magnitudes for densification.

        Args:
            radii:     (N,) screen radii.
            grad_norm: (N,) L2 norm of 2-D screen-space gradient per Gaussian.
        """
        n      = self._xyz.shape[0]
        device = self._xyz.device

        # Move accumulators to correct device if needed
        for attr in ("xyz_gradient_accum", "denom", "max_radii2D"):
            t = getattr(self, attr)
            if t.device != device:
                setattr(self, attr, t.to(device))

        # Align lengths — safety padding/truncation
        if radii.shape[0] != n or grad_norm.shape[0] != n:
            r = torch.zeros(n, device=device)
            g = torch.zeros(n, device=device)
            m = min(radii.shape[0], n)
            r[:m] = radii[:m].float().to(device)
            g[:m] = grad_norm[:m].to(device)
            radii, grad_norm = r, g
        else:
            radii     = radii.float().to(device)
            grad_norm = grad_norm.to(device)

        visible = radii > 0
        if not visible.any():
            return

        self.xyz_gradient_accum[visible] += grad_norm[visible].unsqueeze(-1)
        self.denom[visible]              += 1
        self.max_radii2D[visible]         = torch.max(
            self.max_radii2D[visible], radii[visible]
        )

    # ------------------------------------------------------------------
    # Densification entry point
    # ------------------------------------------------------------------

    def reset_opacity(self, optimizer=None, value: float = 0.01) -> None:
        """
        Reset all opacities to a low value (standard 3DGS anti-floater trick).

        Periodically forcing every Gaussian to become near-transparent lets
        gradient descent "re-earn" opacity only for Gaussians that genuinely
        improve the render, pruning away needle/floater Gaussians that had
        drifted to high opacity without contributing real detail.

        A small per-Gaussian logit jitter is applied after the reset so Adam
        does not see every Gaussian as identical (which would otherwise kill
        per-Gaussian gradient differentiation and stall appearance learning).
        """
        with torch.no_grad():
            target = _inverse_sigmoid(
                torch.full_like(self._opacities, value)
            )
            self._opacities.data.copy_(target)
            # ±0.05 logit jitter — keeps opacities near the floor while giving
            # the optimiser distinct starting points per Gaussian.
            jitter = (torch.rand_like(self._opacities) - 0.5) * 0.1
            self._opacities.data.add_(jitter)

        if optimizer is not None:
            for group in optimizer.param_groups:
                if group.get("name") == "opacity" and len(group["params"]) == 1:
                    p = group["params"][0]
                    state = optimizer.state.get(p, {})
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].zero_()

    def densify_and_prune(
        self,
        max_grad:        float,
        min_opacity:     float,
        extent:          float,
        max_screen_size: float,
        optimizer=None,
    ) -> None:
        """Clone/split high-gradient Gaussians and prune transparent ones."""
        # Snapshot pre-clone count so split indexes only original rows
        n_orig = self._xyz.shape[0]
        grads  = self.xyz_gradient_accum / self.denom.clamp_min(1)
        grads[grads.isnan()] = 0.0

        self._densify_and_clone(grads, max_grad, extent, optimizer)
        self._densify_and_split(grads, max_grad, extent, optimizer, n_orig=n_orig)

        # Prune: opacity floor + world-size gate (always) + screen-size gate (optional)
        # [BIG-WS-FIX-1] big_ws (oversized Gaussian prune) is now UNCONDITIONAL.
        # Previously it was gated behind max_screen_size > 0, but all training stages
        # set max_screen=0 — so oversized Gaussians (the bokeh blobs) were NEVER pruned.
        # big_vs (screen-pixel radii) remains optional, only used when max_screen_size > 0.
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        big_ws = self.get_scaling.max(dim=1).values > 0.5 * extent   # always applied
        prune_mask = prune_mask | big_ws
        if max_screen_size > 0:
            big_vs     = self.max_radii2D > max_screen_size           # screen-size only when requested
            prune_mask = prune_mask | big_vs

        self._prune_points(prune_mask, optimizer)

        # Reset accumulators
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Clone — small Gaussians with high gradient
    # ------------------------------------------------------------------

    def _densify_and_clone(
        self, grads, grad_threshold, scene_extent, optimizer=None
    ) -> None:
        selected = (
            (grads.squeeze(-1) >= grad_threshold)
            & (self.get_scaling.max(dim=1).values <= 0.1 * scene_extent)
        )
        if not selected.any():
            return

        new_tensors = {
            "xyz":      self._xyz[selected].detach(),
            "f_dc":     self._features_dc[selected].detach(),
            "f_rest":   self._features_rest[selected].detach(),
            "opacity":  self._opacities[selected].detach(),
            "scaling":  self._scales[selected].detach(),
            "rotation": self._rotations[selected].detach(),
        }
        self._append_gaussians(new_tensors, optimizer)

    # ------------------------------------------------------------------
    # Split — large Gaussians with high gradient
    # ------------------------------------------------------------------

    def _densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        optimizer=None,
        N: int = 2,
        n_orig: int = -1,
    ) -> None:
        n = n_orig if n_orig > 0 else grads.shape[0]
        grads_orig = grads[:n]

        selected = torch.zeros(n, dtype=torch.bool, device=self._xyz.device)
        mask = (
            (grads_orig.squeeze(-1) >= grad_threshold)
            & (self.get_scaling[:n].max(dim=1).values > 0.1 * scene_extent)
        )
        selected[:len(mask)] = mask
        if not selected.any():
            return

        scales  = self.get_scaling[selected].repeat(N, 1) / (0.8 * N)
        rots    = self.get_rotation[selected].repeat(N, 1)
        samples = torch.normal(mean=torch.zeros_like(scales), std=scales)
        R_mats  = _build_rotation(rots)
        new_xyz = (R_mats @ samples.unsqueeze(-1)).squeeze(-1) \
                  + self._xyz[selected].repeat(N, 1)

        new_scales = torch.log(scales).clamp(-7.0, self._max_log_scale)

        new_tensors = {
            "xyz":      new_xyz,
            "f_dc":     self._features_dc[selected].repeat(N, 1, 1),
            "f_rest":   self._features_rest[selected].repeat(N, 1, 1),
            "opacity":  self._opacities[selected].repeat(N, 1),
            "scaling":  new_scales,
            "rotation": rots,
        }
        self._append_gaussians(new_tensors, optimizer)

        # Remove original Gaussians that were split
        n_total        = self._xyz.shape[0]
        prune_selected = torch.zeros(n_total, dtype=torch.bool, device=self._xyz.device)
        prune_selected[:n] = selected
        self._prune_points(prune_selected, optimizer)

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def _prune_points(self, mask: Tensor, optimizer=None) -> None:
        """Remove Gaussians where mask is True; patch optimizer state."""
        if mask.numel() == 0:
            return

        mask = mask.to(device=self._xyz.device, dtype=torch.bool).view(-1)

        # Align mask length
        if mask.shape[0] != self._xyz.shape[0]:
            fixed = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            m = min(mask.shape[0], fixed.shape[0])
            fixed[:m] = mask[:m]
            mask = fixed

        keep      = ~mask
        min_keep  = min(10000, self._xyz.shape[0])   # hard floor — never collapse below 10k

        if int(keep.sum().item()) < min_keep:
            opacity   = self.get_opacity.detach().squeeze(-1)
            _, idx    = torch.topk(opacity, k=min_keep, largest=True)
            keep      = torch.zeros_like(mask)
            keep[idx] = True

        if optimizer is not None:
            _prune_optimizer_states(optimizer, keep)

        with torch.no_grad():
            self._xyz           = nn.Parameter(self._xyz[keep])
            self._features_dc   = nn.Parameter(self._features_dc[keep])
            self._features_rest = nn.Parameter(self._features_rest[keep])
            self._opacities     = nn.Parameter(self._opacities[keep])
            self._scales        = nn.Parameter(self._scales[keep])
            self._rotations     = nn.Parameter(self._rotations[keep])

        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom               = self.denom[keep]
        self.max_radii2D         = self.max_radii2D[keep]

        if optimizer is not None:
            _reassign_optimizer_params(optimizer, self)

    # ------------------------------------------------------------------
    # Append (used by clone and split)
    # ------------------------------------------------------------------

    def _append_gaussians(self, new_tensors: dict, optimizer=None) -> None:
        n_new  = new_tensors["xyz"].shape[0]
        device = self._xyz.device

        if optimizer is not None:
            _extend_optimizer_states(optimizer, new_tensors)

        with torch.no_grad():
            self._xyz           = nn.Parameter(torch.cat([self._xyz,           new_tensors["xyz"]],      dim=0))
            self._features_dc   = nn.Parameter(torch.cat([self._features_dc,   new_tensors["f_dc"]],     dim=0))
            self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_tensors["f_rest"]],   dim=0))
            self._opacities     = nn.Parameter(torch.cat([self._opacities,     new_tensors["opacity"]],  dim=0))
            self._scales        = nn.Parameter(torch.cat([self._scales,        new_tensors["scaling"]],  dim=0))
            self._rotations     = nn.Parameter(torch.cat([self._rotations,     new_tensors["rotation"]], dim=0))

        zeros_1 = torch.zeros(n_new, 1, device=device)
        zeros_0 = torch.zeros(n_new,    device=device)
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, zeros_1], dim=0)
        self.denom               = torch.cat([self.denom,               zeros_1], dim=0)
        self.max_radii2D         = torch.cat([self.max_radii2D,         zeros_0], dim=0)

        if optimizer is not None:
            _reassign_optimizer_params(optimizer, self)


# ---------------------------------------------------------------------------
# Optimizer state helpers
# ---------------------------------------------------------------------------

# Maps optimizer param-group "name" → GaussianModel attribute name
_GROUP_TO_ATTR = {
    "xyz":      "_xyz",
    "f_dc":     "_features_dc",
    "f_rest":   "_features_rest",
    "opacity":  "_opacities",
    "scaling":  "_scales",
    "rotation": "_rotations",
}


def _prune_optimizer_states(optimizer, keep: Tensor) -> None:
    for group in optimizer.param_groups:
        if group.get("name", "") not in _GROUP_TO_ATTR or len(group["params"]) != 1:
            continue
        p     = group["params"][0]
        state = optimizer.state.get(p, {})
        if "exp_avg" in state:
            state["exp_avg"]    = state["exp_avg"][keep]
            state["exp_avg_sq"] = state["exp_avg_sq"][keep]
            optimizer.state[p]  = state


def _extend_optimizer_states(optimizer, new_tensors: dict) -> None:
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name not in _GROUP_TO_ATTR or name not in new_tensors or len(group["params"]) != 1:
            continue
        p     = group["params"][0]
        state = optimizer.state.get(p, {})
        ext   = new_tensors[name]
        if "exp_avg" in state:
            state["exp_avg"]    = torch.cat([state["exp_avg"],    torch.zeros_like(ext)], dim=0)
            state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(ext)], dim=0)
            optimizer.state[p]  = state


def _reassign_optimizer_params(optimizer, model) -> None:
    """After in-place parameter replacement, point each group at the new tensor."""
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name not in _GROUP_TO_ATTR or len(group["params"]) != 1:
            continue
        old_p              = group["params"][0]
        new_p              = getattr(model, _GROUP_TO_ATTR[name])
        state              = optimizer.state.pop(old_p, {})
        optimizer.state[new_p] = state
        group["params"][0] = new_p


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1.0 - x))


def _build_rotation(r: Tensor) -> Tensor:
    norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r    = r / norm
    w, x, y, z = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=1),
    ], dim=1)


def _dist_cpu_chunked(xyz: Tensor, chunk_size: int = 4096) -> Tensor:
    """Nearest-neighbour squared distance, O(N·chunk) memory (CPU fallback)."""
    n         = xyz.shape[0]
    min_dists = torch.full((n,), float("inf"))
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        dists = torch.cdist(xyz[start:end], xyz)
        for local_i in range(end - start):
            dists[local_i, start + local_i] = float("inf")
        min_dists[start:end] = dists.min(dim=1).values
    return min_dists


def _dist_gpu_chunked(xyz: Tensor, chunk_q: int = 8_192, chunk_r: int = 8_192) -> Tensor:
    """
    Correct self-KNN on GPU via double-chunked pairwise distances.

    Used when simple_knn is unavailable or N exceeds its safe limit.
    Memory per step: chunk_q × chunk_r × 4 bytes ≈ 256 MB at defaults —
    well within T4/A100 headroom after the point cloud is loaded.

    Time: O(ceil(N/chunk_q) × ceil(N/chunk_r)) cdist calls.
    For N=100k, chunk=8k: ~13×13 = 169 steps, typically < 10 s on T4.
    """
    n      = xyz.shape[0]
    dev    = xyz.device
    min_d2 = torch.full((n,), float("inf"), device=dev)

    for q_start in range(0, n, chunk_q):
        q_end = min(q_start + chunk_q, n)
        q     = xyz[q_start:q_end]           # (cq, 3)

        for r_start in range(0, n, chunk_r):
            r_end = min(r_start + chunk_r, n)
            r     = xyz[r_start:r_end]        # (cr, 3)

            d2 = torch.cdist(q, r).pow(2)     # (cq, cr)

            # Mask self-distances where query and reference windows overlap
            if q_start < r_end and r_start < q_end:
                ov_q = torch.arange(
                    max(q_start, r_start) - q_start,
                    min(q_end,   r_end)   - q_start,
                    device=dev,
                )
                ov_r = ov_q + (max(q_start, r_start) - r_start)
                d2[ov_q, ov_r] = float("inf")

            chunk_min = d2.min(dim=1).values
            min_d2[q_start:q_end] = torch.minimum(min_d2[q_start:q_end], chunk_min)

    return min_d2


# simple_knn's distCUDA2 uses 32-bit tile arithmetic that overflows and writes
# out of bounds (cudaErrorIllegalAddress / SIGABRT) when N exceeds ~65k points.
# Points above this threshold are routed through _dist_gpu_chunked instead.
_SIMPLE_KNN_SAFE_N = 65_000


def _distCUDA2(xyz: Tensor) -> Tensor:
    """
    Nearest-neighbour squared distance.

    Priority:
      1. simple_knn.distCUDA2  — fast, but only safe for N ≤ _SIMPLE_KNN_SAFE_N
      2. _dist_gpu_chunked     — correct for any N, runs on CUDA if available
      3. _dist_cpu_chunked     — CPU fallback when no CUDA device is present
    """
    n = xyz.shape[0]
    try:
        from simple_knn._C import distCUDA2  # type: ignore
        if n <= _SIMPLE_KNN_SAFE_N:
            return distCUDA2(xyz)
        # N is too large for simple_knn — fall through to chunked GPU path
    except ImportError:
        pass

    if xyz.is_cuda:
        return _dist_gpu_chunked(xyz)
    return _dist_cpu_chunked(xyz.cpu()).to(xyz.device)