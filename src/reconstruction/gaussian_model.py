"""
gaussian_model.py — 3D Gaussian representation.

FIXES APPLIED:
  A. Scale instability:    clamp range narrowed (-6,2) → (-4,1)
  B. Quaternion normalise: eps=1e-8 to avoid zero-division
  C. Opacity instability:  clamp sigmoid output to (1e-4, 0.9999)
  D. OOM in _dist_cpu:     chunked KD-tree nearest-neighbour (O(N·chunk))
  E. SH-DC coefficient:    was storing raw RGB; must store (rgb-0.5)/SH_C0
                           so that _eval_sh (which adds back 0.5) round-trips.
  F. Gradient accumulators: xyz_gradient_accum, denom, max_radii2D now
                            initialised in initialise_from_pcd so that
                            densify_and_prune doesn't crash at iter 500.
  G. Densification stubs:  _densify_and_clone, _densify_and_split, and
                            _prune_points fully implemented with optimizer
                            state patching (Adam exp_avg / exp_avg_sq).
  H. reset_opacity:        implemented (called by Trainer every 1000 iters).

BLUR FIXES APPLIED:
  BF-1. get_scaling upper clamp: 2.0 → 1.0
        Matches the initial scale clamp (max=0.0 log-space = 1.0 linear).
        Prevents runaway Gaussian growth post-densification while still
        allowing meaningful surface coverage.
  BF-2. Initial opacity: 0.1 → 0.5
        Very low initial opacity starved early training of gradient signal.
        Higher initial opacity forces correct Gaussian placement from the start.
  BF-3. reset_opacity floor: 0.01 → 0.05
        Preserves enough accumulated structure between resets.
  BF-4. _densify_and_split scale clamp: 2.0 → 1.0 (matches BF-1).

PATCH APPLIED (June 2026):
  P-1. Initial log-scales clamped to [-4, 0] in initialise_from_pcd.
       Previously unclamped values could exceed 0.0 (i.e. Gaussian diameter > 1
       world-unit) when COLMAP point spacing is large, producing giant blobs
       that dominate early training and never recover.  Forcing max=0.0 means
       all initial Gaussians are ≤ 1 world-unit and must earn their size.

FIX APPLIED (June 2026 — foggy preview):
  FP-1. initialise_from_pcd now computes max_log_scale from spatial_lr_scale
        (= cameras_extent, passed by train.py after scene normalization).
        After normalize_scene the scene fits inside a ~0.047-radius sphere,
        so clamping log-scales to 0.0 still produces Gaussians with diameter
        exp(0) = 1.0 world-unit — 21× larger than the entire scene.  Every
        Gaussian is a giant translucent blob and the render is solid grey fog.
        Fix: max_log_scale = log(cameras_extent * 0.1), which keeps each
        initial Gaussian at ~10% of the scene radius.  After normalization
        cameras_extent is floored to 1.0 in train.py (see FP-2 there), so
        the default behaviour (spatial_lr_scale=1.0) is max_log_scale = log(0.1)
        ≈ −2.3, giving initial Gaussian diameter ≈ 0.1 world-units — correct
        for a unit-sphere scene.
  FP-2. get_scaling upper clamp raised from 1.0 to exp(0) = 1.0 but now driven
        by self._max_log_scale stored during initialise_from_pcd, so cloning/
        splitting during densification cannot grow Gaussians beyond the same
        ceiling that was used at init time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

SH_C0 = 0.28209479177387814   # matches renderer._eval_sh constant


class GaussianModel(nn.Module):
    """Learnable set of 3-D Gaussians used for splatting-based rendering."""

    def __init__(self, sh_degree: int = 3) -> None:
        super().__init__()
        self.sh_degree       = sh_degree
        self.active_sh_degree = 0
        self.max_sh_degree   = sh_degree

        # Learnable parameters (populated by initialise_from_pcd or checkpoint)
        self._xyz:           Tensor = torch.empty(0)
        self._features_dc:   Tensor = torch.empty(0)
        self._features_rest: Tensor = torch.empty(0)
        self._scales:        Tensor = torch.empty(0)   # log-space
        self._rotations:     Tensor = torch.empty(0)   # quaternion (unnorm.)
        self._opacities:     Tensor = torch.empty(0)   # logit-space

        # Gradient accumulation tensors (initialised after initialise_from_pcd)
        self.xyz_gradient_accum: Tensor = torch.empty(0)
        self.denom:              Tensor = torch.empty(0)
        self.max_radii2D:        Tensor = torch.empty(0)

        # FP-2: scale ceiling — set by initialise_from_pcd, used by get_scaling
        # and _densify_and_split so cloned/split Gaussians obey the same limit.
        self._max_log_scale: float = 0.0

    # ------------------------------------------------------------------
    # Properties — "activated" values used during rendering
    # ------------------------------------------------------------------

    @property
    def get_xyz(self) -> Tensor:
        return self._xyz

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians currently in the model."""
        return self._xyz.shape[0]

    @property
    def get_features(self) -> Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_scaling(self) -> Tensor:
        # FP-2: upper clamp tracks self._max_log_scale (set during
        # initialise_from_pcd from cameras_extent) so Gaussians cannot grow
        # beyond the same ceiling used at initialisation time.
        # Lower floor: -4 = exp(-4) ≈ 0.018 world-units minimum.
        return torch.exp(torch.clamp(self._scales, min=-4.0, max=self._max_log_scale))

    @property
    def get_rotation(self) -> Tensor:
        # FIX B: eps to prevent NaN when quaternion norm ≈ 0
        return F.normalize(self._rotations, dim=-1, eps=1e-8)

    @property
    def get_opacity(self) -> Tensor:
        # FIX C: clamp so opacity never reaches exact 0 or 1
        return torch.sigmoid(self._opacities).clamp(1e-4, 0.9999)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._xyz.shape[0]

    def to(self, *args, **kwargs):
        """Move parameters and non-parameter training accumulators together."""
        module = super().to(*args, **kwargs)
        device = self._xyz.device
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
        self.denom = self.denom.to(device)
        self.max_radii2D = self.max_radii2D.to(device)
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
                "sh_dc": self._features_dc.detach().cpu().numpy(),
                "sh_rest": self._features_rest.detach().cpu().numpy(),
                "colors": colors.detach().cpu().numpy(),
                "opacities": self.get_opacity.detach().cpu().numpy(),
                "scales": self.get_scaling.detach().cpu().numpy(),
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
        """Populate model parameters from a coloured point cloud.

        Args:
            xyz:              (N, 3) point positions in normalized scene space.
            rgb:              (N, 3) colours in [0, 1].
            spatial_lr_scale: cameras_extent after normalization (floored to 1.0
                              in train.py).  Used to derive the initial Gaussian
                              size ceiling: each Gaussian starts at ≤10% of the
                              scene radius rather than at an absolute 1.0 world-
                              unit, which was 21× too large for a normalized scene.
        """
        self.spatial_lr_scale = spatial_lr_scale

        # FP-1: compute scale ceiling relative to scene extent so Gaussians
        # start small enough to resolve detail in the normalized scene.
        # cameras_extent is floored to 1.0 in train.py (FP-2 there), so the
        # default path (no normalization) gives max_log_scale = log(0.1) ≈ −2.3.
        # For a typical normalized scene (extent ≈ 1.0) each Gaussian starts
        # with diameter exp(−2.3) ≈ 0.1 world-units — correct for a unit sphere.
        self._max_log_scale = math.log(max(spatial_lr_scale * 0.1, 1e-4))

        n = xyz.shape[0]

        # FIX E: convert raw RGB [0,1] → SH-DC coefficient so that
        #        _eval_sh(SH_C0 * sh_dc + 0.5) == rgb.
        sh_dc   = ((rgb - 0.5) / SH_C0).unsqueeze(1)          # (N,1,3)
        sh_rest = torch.zeros(n, (self.max_sh_degree + 1) ** 2 - 1, 3)

        # FIX D: chunked nearest-neighbour instead of O(N²) cdist
        # - _distCUDA2 returns squared L2 distances
        # - _dist_cpu_chunked returns L2 distances (not squared)
        if xyz.is_cuda:
            dist_sq = _distCUDA2(xyz)
            dist = torch.sqrt(dist_sq.clamp_min(1e-7))
        else:
            dist = _dist_cpu_chunked(xyz).clamp_min(1e-7)

        scales = torch.log(dist).unsqueeze(-1).repeat(1, 3)


        # FP-1: clamp initial log-scales to [-4, _max_log_scale].
        # Old hard ceiling was 0.0 (exp(0) = 1.0 world-unit).  After
        # normalize_scene the scene is ~0.047 units wide, so a 1.0-unit
        # Gaussian covers the entire scene — hence solid grey fog at iter 500.
        scales = torch.clamp(scales, min=-4.0, max=self._max_log_scale)

        rots = torch.zeros(n, 4)
        rots[:, 0] = 1.0                                       # identity quat

        # BLUR FIX: raised initial opacity from 0.1 → 0.5.
        # Low initial opacity (0.1) means Gaussians are nearly transparent at
        # the start of training. The optimizer keeps them transparent because
        # the gradient signal is weak, leading to a blurry underfit result.
        opacities = _inverse_sigmoid(0.5 * torch.ones(n, 1))

        self._xyz           = nn.Parameter(xyz.float())
        self._features_dc   = nn.Parameter(sh_dc.float().contiguous())
        self._features_rest = nn.Parameter(sh_rest.float().contiguous())
        self._scales        = nn.Parameter(scales.float())
        self._rotations     = nn.Parameter(rots.float())
        self._opacities     = nn.Parameter(opacities.float())

        # FIX F: initialise gradient-accumulation tensors
        device = xyz.device
        self.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        self.denom              = torch.zeros(n, 1, device=device)
        self.max_radii2D        = torch.zeros(n,    device=device)

    # ------------------------------------------------------------------
    # Opacity reset (called by Trainer every opacity_reset_interval iters)
    # ------------------------------------------------------------------

    def reset_opacity(self) -> None:   # BLUR FIX: raised reset floor from 0.01 → 0.05
        """Reset all opacities to a low-but-visible value (logit of 0.05).

        BLUR FIX: The original 0.01 reset was so low that Gaussians became
        nearly invisible after each reset, forcing the optimizer to rebuild
        coverage from scratch every 3000 iterations. This starves the scene
        of accumulated opacity structure and causes blurry renders.
        Raising to 0.05 preserves enough signal while still allowing
        transparent floaters to be pruned.
        """
        with torch.no_grad():
            self._opacities.data.fill_(_inverse_sigmoid(
                torch.tensor(0.05)
            ).item())

    # ------------------------------------------------------------------
    # Gradient-accumulation update (called by Trainer after each render)
    # ------------------------------------------------------------------

    def update_stats(self, radii: Tensor, viewspace_grad: Tensor) -> None:
        """
        Accumulate screen-space gradient norms for densification decisions.
        Legacy API: takes dense (N,) radii and (N, 2) viewspace gradients.
        """
        grad_norm = viewspace_grad.norm(dim=-1)  # (N,)
        self.update_stats_norm(radii, grad_norm)

    def update_stats_norm(self, radii: Tensor, grad_norm: Tensor) -> None:
        """
        Accumulate pre-normed gradient magnitudes for densification decisions.

        Args:
            radii:     (N,) — screen radii (float or int). Dense, same length as _xyz.
            grad_norm: (N,) — L2 norm of 2-D screen-space gradient per Gaussian.

        This is the preferred API when called from packed-mode gsplat output
        (where Trainer.update_gradient_accum has already scattered sparse→dense).
        """
        n = self._xyz.shape[0]
        device = self._xyz.device

        if self.xyz_gradient_accum.device != device:
            self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
        if self.denom.device != device:
            self.denom = self.denom.to(device)
        if self.max_radii2D.device != device:
            self.max_radii2D = self.max_radii2D.to(device)

        if radii.shape[0] != n or grad_norm.shape[0] != n:
            # Safety: truncate/zero-pad to match current model size
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
    # Densification — FIX G: full implementation with optimizer patching
    # ------------------------------------------------------------------

    def densify_and_prune(
        self,
        max_grad:        float,
        min_opacity:     float,
        extent:          float,
        max_screen_size: float,
        optimizer=None,
    ) -> None:
        """Clone/split high-gradient Gaussians and prune transparent ones."""
        grads = self.xyz_gradient_accum / self.denom.clamp_min(1)
        grads[grads.isnan()] = 0.0

        self._densify_and_clone(grads, max_grad, extent, optimizer)
        self._densify_and_split(grads, max_grad, extent, optimizer)

        # [PRUNE-FIX] big_ws MUST live inside the max_screen_size > 0 gate.
        # The broken variant (threshold 0.5*extent, unconditional) wiped
        # ~19 k Gaussians at iter 1600 on every run — the first step where
        # max_screen_size switches from 0 → 20.  Correct behaviour:
        #   • big_ws only fires together with big_vs (screen-size warmup guard)
        #   • threshold is 0.1*extent, not 0.5*extent
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size > 0:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_vs | big_ws

        self._prune_points(prune_mask, optimizer)
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
        torch.cuda.empty_cache()

    def _densify_and_clone(
        self, grads, grad_threshold, scene_extent, optimizer=None
    ) -> None:
        """Clone small-scale Gaussians that have high screen-space gradient."""
        selected = (
            (grads.squeeze(-1) >= grad_threshold)
            & (self.get_scaling.max(dim=1).values <= 0.01 * scene_extent)
        )
        if not selected.any():
            return

        new_xyz       = self._xyz[selected].detach()
        new_features_dc   = self._features_dc[selected].detach()
        new_features_rest = self._features_rest[selected].detach()
        new_opacities = self._opacities[selected].detach()
        new_scales    = self._scales[selected].detach()
        new_rots      = self._rotations[selected].detach()

        new_tensors = {
            "xyz":    new_xyz,
            "f_dc":   new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scales,
            "rotation": new_rots,
        }
        self._append_gaussians(new_tensors, optimizer)

    def _densify_and_split(
        self, grads, grad_threshold, scene_extent, optimizer=None, N: int = 2
    ) -> None:
        """Split large-scale Gaussians that have high screen-space gradient."""
        n = self._xyz.shape[0]
        selected = torch.zeros(n, dtype=torch.bool, device=self._xyz.device)
        selected_mask = (
            (grads.squeeze(-1) >= grad_threshold)
            & (self.get_scaling.max(dim=1).values > 0.01 * scene_extent)
        )
        selected[:len(selected_mask)] = selected_mask
        if not selected.any():
            return

        scales  = self.get_scaling[selected].repeat(N, 1) / (0.8 * N)
        rots    = self.get_rotation[selected].repeat(N, 1)
        means   = torch.zeros_like(scales)
        samples = torch.normal(mean=means, std=scales)
        R_mats  = _build_rotation(rots)
        new_xyz = (R_mats @ samples.unsqueeze(-1)).squeeze(-1) \
                  + self._xyz[selected].repeat(N, 1)
        new_scales    = torch.log(scales).clamp(-4.0, self._max_log_scale)  # FP-2: track init ceiling
        new_features_dc   = self._features_dc[selected].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacities = self._opacities[selected].repeat(N, 1)
        new_rots      = rots

        new_tensors = {
            "xyz":    new_xyz,
            "f_dc":   new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scales,
            "rotation": new_rots,
        }
        self._append_gaussians(new_tensors, optimizer)

        # Remove the originals that were split
        prune_selected = torch.cat([
            selected,
            torch.zeros(N * selected.sum(), dtype=torch.bool, device=self._xyz.device),
        ])
        self._prune_points(prune_selected, optimizer)

    def _prune_points(self, mask: Tensor, optimizer=None) -> None:
        """Remove Gaussians where mask is True; patch optimizer state."""
        if mask.numel() == 0:
            return

        mask = mask.to(device=self._xyz.device, dtype=torch.bool).view(-1)
        if mask.shape[0] != self._xyz.shape[0]:
            fixed = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
            n = min(mask.shape[0], fixed.shape[0])
            fixed[:n] = mask[:n]
            mask = fixed

        keep = ~mask
        min_keep = min(10000, self._xyz.shape[0])  # [COLLAPSE-FIX] was 1000 — floor raised to 10k

        if int(keep.sum().item()) < min_keep:
            opacity = self.get_opacity.detach().squeeze(-1)
            _, idx = torch.topk(opacity, k=min_keep, largest=True)
            keep = torch.zeros_like(mask)
            keep[idx] = True

        if optimizer is not None:
            _prune_optimizer_states(optimizer, self, keep)

        with torch.no_grad():
            self._xyz           = nn.Parameter(self._xyz[keep])
            self._features_dc   = nn.Parameter(self._features_dc[keep])
            self._features_rest = nn.Parameter(self._features_rest[keep])
            self._opacities     = nn.Parameter(self._opacities[keep])
            self._scales        = nn.Parameter(self._scales[keep])
            self._rotations     = nn.Parameter(self._rotations[keep])

        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom              = self.denom[keep]
        self.max_radii2D        = self.max_radii2D[keep]

        # Rebuild optimizer param_groups to reference new parameters
        if optimizer is not None:
            _reassign_optimizer_params(optimizer, self)

    def _append_gaussians(self, new_tensors: dict, optimizer=None) -> None:
        """
        Append new Gaussians (from clone/split) and extend optimizer state.
        new_tensors keys: "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"
        """
        n_new = new_tensors["xyz"].shape[0]
        device = self._xyz.device

        if optimizer is not None:
            _extend_optimizer_states(optimizer, self, new_tensors)

        with torch.no_grad():
            self._xyz = nn.Parameter(
                torch.cat([self._xyz, new_tensors["xyz"]], dim=0))
            self._features_dc = nn.Parameter(
                torch.cat([self._features_dc, new_tensors["f_dc"]], dim=0))
            self._features_rest = nn.Parameter(
                torch.cat([self._features_rest, new_tensors["f_rest"]], dim=0))
            self._opacities = nn.Parameter(
                torch.cat([self._opacities, new_tensors["opacity"]], dim=0))
            self._scales = nn.Parameter(
                torch.cat([self._scales, new_tensors["scaling"]], dim=0))
            self._rotations = nn.Parameter(
                torch.cat([self._rotations, new_tensors["rotation"]], dim=0))

        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros(n_new, 1, device=device)], dim=0)
        self.denom = torch.cat(
            [self.denom, torch.zeros(n_new, 1, device=device)], dim=0)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros(n_new, device=device)], dim=0)

        if optimizer is not None:
            _reassign_optimizer_params(optimizer, self)


# ---------------------------------------------------------------------------
# Optimizer state helpers (used by densification)
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


def _prune_optimizer_states(optimizer, model, keep: Tensor) -> None:
    """Prune Adam exp_avg / exp_avg_sq tensors to match `keep` mask."""
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name not in _GROUP_TO_ATTR or len(group["params"]) != 1:
            continue
        p = group["params"][0]
        state = optimizer.state.get(p, {})
        if "exp_avg" in state:
            state["exp_avg"]    = state["exp_avg"][keep]
            state["exp_avg_sq"] = state["exp_avg_sq"][keep]
            optimizer.state[p]  = state


def _extend_optimizer_states(optimizer, model, new_tensors: dict) -> None:
    """Extend Adam exp_avg / exp_avg_sq tensors with zeros for new Gaussians."""
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
    """
    After in-place parameter replacement (clone/split/prune), point each
    optimizer param_group at the model's current nn.Parameter tensor.
    """
    for group in optimizer.param_groups:
        name = group.get("name", "")
        if name not in _GROUP_TO_ATTR or len(group["params"]) != 1:
            continue
        old_p   = group["params"][0]
        new_p   = getattr(model, _GROUP_TO_ATTR[name])
        state   = optimizer.state.pop(old_p, {})
        optimizer.state[new_p] = state
        group["params"][0]     = new_p


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1 - x))


def _build_rotation(r: Tensor) -> Tensor:
    """Build rotation matrices from (N, 4) quaternions."""
    norm = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r    = r / norm
    w, x, y, z = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=1),
    ], dim=1)
    return R


def _dist_cpu_chunked(xyz: Tensor, chunk_size: int = 4096) -> Tensor:
    """
    Nearest-neighbour squared distance — O(N · chunk_size) memory.
    FIX D: replaces the O(N²) torch.cdist call that OOM'd on T4.
    """
    n = xyz.shape[0]
    min_dists = torch.full((n,), float("inf"))
    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        chunk_dists = torch.cdist(xyz[start:end], xyz)
        for local_i in range(end - start):
            chunk_dists[local_i, start + local_i] = float("inf")
        min_dists[start:end] = chunk_dists.min(dim=1).values
    return min_dists


def _distCUDA2(xyz: Tensor) -> Tensor:
    """Nearest-neighbour squared distance (CUDA fast path via simple_knn)."""
    try:
        from simple_knn._C import distCUDA2  # type: ignore
        return distCUDA2(xyz)
    except ImportError:
        return _dist_cpu_chunked(xyz.cpu()).to(xyz.device)