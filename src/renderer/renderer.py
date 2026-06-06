"""
src/renderer/renderer.py
-------------------------
Gaussian Splatting renderer using diff-gaussian-rasterization.

Replaces the previous gsplat-based renderer entirely.

WHY THIS CHANGE:
  The gsplat renderer required absgrad=True + packed=True to get screen-space
  gradient signals for densification. In practice, the absgrad tensor was not
  attaching correctly to the backward graph (meta["means2d"].absgrad was always
  None or zeros), causing xyz_gradient_accum to never grow → delta=+0 at every
  densification step → Gaussian count frozen → blurry output forever.

  diff-gaussian-rasterization (the original 3DGS CUDA kernel) writes gradients
  directly to viewspace_point_tensor.grad via standard autograd — no special
  absgrad hook, no packed mode scatter, no gaussian_ids mapping. This is the
  battle-tested path used in the original paper.

INTERFACE CONTRACT (unchanged — colab/train.py and trainer.py call these):
  renderer.render_torch(model, camera) → (rendered_image, meta)
    rendered_image : (3, H, W) float32 tensor in [0, 1]
    meta           : dict with keys:
                       "viewspace_points" : (N, 3) tensor with .grad after backward
                       "visibility_filter": (N,)  bool tensor
                       "radii"            : (N,)  int tensor

  GaussianRenderer.__init__(device="cuda")
  GaussianRenderer.render_torch(model, camera)
  GaussianRenderer.render(model, camera) → (H, W, 3) uint8 numpy

CAMERA CONTRACT (camera.py is unchanged):
  camera.FoVx, camera.FoVy     : float (radians)
  camera.image_width, .image_height : int
  camera.world_view_transform  : (4, 4) numpy float32
  camera.full_proj_transform   : (4, 4) numpy float32
  camera.position              : (3,)  numpy float32

GAUSSIAN MODEL CONTRACT (gaussian_model.py is replaced with original 3DGS):
  model.get_xyz          : (N, 3)
  model.get_features     : (N, K, 3) SH coefficients
  model.get_opacity      : (N, 1) after sigmoid
  model.get_scaling      : (N, 3) after exp
  model.get_rotation     : (N, 4) normalised quaternion
  model.active_sh_degree : int
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .camera import Camera


# ---------------------------------------------------------------------------
# diff-gaussian-rasterization import (required — install via submodule)
# ---------------------------------------------------------------------------

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    _DIFF_GAUSS = True
except ImportError:
    _DIFF_GAUSS = False


# ---------------------------------------------------------------------------
# SH utilities (copied from gaussian-splatting/utils/sh_utils.py)
# Used by both the main renderer and the software fallback.
# ---------------------------------------------------------------------------

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.4453057213903705,
          -0.5900435899266435]


def _eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH up to given degree. sh: (N, K, 3), dirs: (N, 3)."""
    result = SH_C0 * sh[:, 0, :]
    if degree < 1:
        return (result + 0.5).clamp(0.0, 1.0)
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result = result - SH_C1*y*sh[:,1,:] + SH_C1*z*sh[:,2,:] - SH_C1*x*sh[:,3,:]
    if degree < 2:
        return (result + 0.5).clamp(0.0, 1.0)
    xx, yy, zz = x*x, y*y, z*z
    xy, yz, xz = x*y, y*z, x*z
    result = (result
        + SH_C2[0]*xy*sh[:,4,:] + SH_C2[1]*yz*sh[:,5,:]
        + SH_C2[2]*(2*zz-xx-yy)*sh[:,6,:] + SH_C2[3]*xz*sh[:,7,:]
        + SH_C2[4]*(xx-yy)*sh[:,8,:])
    if degree < 3:
        return (result + 0.5).clamp(0.0, 1.0)
    result = (result
        + SH_C3[0]*y*(3*xx-yy)*sh[:,9,:]   + SH_C3[1]*xy*z*sh[:,10,:]
        + SH_C3[2]*y*(4*zz-xx-yy)*sh[:,11,:] + SH_C3[3]*z*(2*zz-3*xx-3*yy)*sh[:,12,:]
        + SH_C3[4]*x*(4*zz-xx-yy)*sh[:,13,:] + SH_C3[5]*z*(xx-yy)*sh[:,14,:]
        + SH_C3[6]*x*(xx-3*yy)*sh[:,15,:])
    return (result + 0.5).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class GaussianRenderer:
    """
    Renders a GaussianModel using diff-gaussian-rasterization (original 3DGS CUDA kernel).

    Args:
        device   : "cuda" or "cpu". CPU falls back to software renderer.
        bg_color : background RGB in [0, 1]. Default black.
    """

    def __init__(
        self,
        width:    int = 800,
        height:   int = 600,
        bg_color: Union[List[float], Tuple[float, ...]] = (0.0, 0.0, 0.0),
        device:   str = "auto",
        batch_size: int = 5000,
        use_gsplat: bool = True,   # ignored — kept for API compat with old renderer
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("[Renderer] ⚠  CUDA requested but not available — falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device

        self.bg_color  = torch.tensor(list(bg_color), dtype=torch.float32, device=self.device)
        self.batch_size = batch_size

        self._use_diff_gauss = _DIFF_GAUSS and self.device == "cuda"

        if self._use_diff_gauss:
            print("[Renderer] ✓ Using diff-gaussian-rasterization (original 3DGS CUDA kernel)")
        else:
            if self.device == "cuda" and not _DIFF_GAUSS:
                print("[Renderer] ⚠  diff-gaussian-rasterization not found — software fallback")
                print("             Install: pip install submodules/diff-gaussian-rasterization")
            else:
                print(f"[Renderer] Software renderer ({self.device.upper()})")

    def render_torch(self, model, camera: Camera) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Render model from camera.

        Returns:
            rendered : (3, H, W) float32 in [0, 1]
            meta     : dict with "viewspace_points", "visibility_filter", "radii"
                       OR None if using software renderer
        """
        if self._use_diff_gauss:
            return self._render_diff_gauss(model, camera)
        return self._render_software(model, camera), None

    def __call__(self, model, camera: Camera):
        return self.render_torch(model, camera)

    def render(self, model, camera: Camera) -> np.ndarray:
        """Render and return (H, W, 3) uint8 numpy array."""
        result = self.render_torch(model, camera)
        t = result[0] if isinstance(result, tuple) else result
        return (t.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # diff-gaussian-rasterization path (primary)
    # ------------------------------------------------------------------

    def _render_diff_gauss(self, model, camera: Camera) -> Tuple[torch.Tensor, Dict]:
        """
        Render using the original 3DGS CUDA rasterizer.

        Gradient flow for densification:
          - viewspace_point_tensor.retain_grad() is called before rasterization
          - After loss.backward(), viewspace_point_tensor.grad contains the
            (N, 3) screen-space gradients used by densify_and_prune
          - Trainer._update_gradient_accum reads meta["viewspace_points"].grad
            and meta["visibility_filter"] — no absgrad, no gaussian_ids scatter

        This is the exact same gradient mechanism as the original gaussian-splatting
        repo. It works reliably because it uses standard autograd, not custom hooks.
        """
        device = self.device

        # ── Camera setup ─────────────────────────────────────────────────────
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)

        bg = self.bg_color.to(device)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=torch.tensor(
                camera.world_view_transform, dtype=torch.float32, device=device
            ),
            projmatrix=torch.tensor(
                camera.full_proj_transform, dtype=torch.float32, device=device
            ),
            sh_degree=model.active_sh_degree,
            campos=torch.tensor(
                camera.position, dtype=torch.float32, device=device
            ),
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # ── Model tensors ─────────────────────────────────────────────────────
        means3D   = model.get_xyz.to(device)
        means2D   = torch.zeros_like(means3D, requires_grad=True)
        means2D.retain_grad()   # CRITICAL: this is where screen-space grads land

        opacity   = model.get_opacity.to(device)
        scales    = model.get_scaling.to(device)
        rotations = model.get_rotation.to(device)

        # SH: get_features is (N, K, 3) — rasterizer expects (N, K, 3) — pass directly
        shs = model.get_features.to(device)

        # ── Rasterize ─────────────────────────────────────────────────────────
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        # rendered_image: (3, H, W) float32
        # radii:          (N,) int32 — 0 means not visible

        meta = {
            "viewspace_points":  means2D,
            "visibility_filter": radii > 0,
            "radii":             radii,
        }

        return rendered_image.clamp(0.0, 1.0), meta

    # ------------------------------------------------------------------
    # Software fallback (CPU / no CUDA kernel)
    # ------------------------------------------------------------------

    def _render_software(self, model, camera: Camera) -> torch.Tensor:
        """Pure-PyTorch software rasterizer. Slow but dependency-free."""
        H, W   = camera.image_height, camera.image_width
        device = self.device

        positions = model.get_xyz.to(device)
        colors_sh = model.get_features.to(device)
        opacities = model.get_opacity.to(device)
        scales    = model.get_scaling.to(device)
        rotations = model.get_rotation.to(device)

        N = positions.shape[0]
        if N == 0:
            return self.bg_color.view(3, 1, 1).expand(3, H, W)

        view   = torch.from_numpy(camera.view_matrix).float().to(device)
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy

        ones  = torch.ones(N, 1, device=device)
        pos_h = torch.cat([positions, ones], dim=1)
        p_cam = (view @ pos_h.T).T
        z_cam = p_cam[:, 2]
        visible = z_cam > 0.01
        if visible.sum() == 0:
            return self.bg_color.view(3, 1, 1).expand(3, H, W)

        positions = positions[visible]; colors_sh = colors_sh[visible]
        opacities = opacities[visible]; scales = scales[visible]
        rotations = rotations[visible]; p_cam = p_cam[visible]; z_cam = z_cam[visible]

        order     = z_cam.argsort(descending=True)
        positions = positions[order]; colors_sh = colors_sh[order]
        opacities = opacities[order].squeeze(1)
        scales    = scales[order];    rotations = rotations[order]; p_cam = p_cam[order]

        cam_pos   = torch.from_numpy(camera.position).float().to(device)
        view_dirs = F.normalize(positions - cam_pos.unsqueeze(0), dim=1)
        colors    = _eval_sh(model.active_sh_degree, colors_sh, view_dirs)

        px = (p_cam[:, 0] / p_cam[:, 2]) * fx + cx
        py = (p_cam[:, 1] / p_cam[:, 2]) * fy + cy

        # Simple isotropic splat (software path is debug-only)
        sigma = scales.mean(dim=1).clamp(min=1e-4)
        radius = (3.0 * sigma * max(fx, fy)).ceil().int().clamp(1, 64)

        transmittance = torch.ones(H, W, device=device)
        accumulated   = torch.zeros(3, H, W, device=device)

        for i in range(len(positions)):
            mu_x, mu_y = px[i].item(), py[i].item()
            r           = radius[i].item()
            alpha_i     = opacities[i]
            x0 = max(0, int(mu_x - r)); x1 = min(W, int(mu_x + r + 1))
            y0 = max(0, int(mu_y - r)); y1 = min(H, int(mu_y + r + 1))
            if x0 >= x1 or y0 >= y1:
                continue
            lx = torch.arange(x0, x1, device=device, dtype=torch.float32) - mu_x
            ly = torch.arange(y0, y1, device=device, dtype=torch.float32) - mu_y
            dy, dx = torch.meshgrid(ly, lx, indexing="ij")
            s2   = (sigma[i] * max(fx, fy)).clamp(min=0.5) ** 2
            maha = (dx*dx + dy*dy) / (2 * s2)
            weight = torch.exp(-maha.clamp(0.0, 20.0)).detach()
            alpha  = (alpha_i * weight).clamp(max=0.9999)
            T_patch = transmittance[y0:y1, x0:x1].clone()
            contrib = alpha * T_patch
            accumulated[:, y0:y1, x0:x1] += contrib.unsqueeze(0) * colors[i].view(3, 1, 1)
            transmittance[y0:y1, x0:x1] = T_patch.detach() * (1.0 - alpha.detach())

        bg = self.bg_color.view(3, 1, 1) * transmittance.unsqueeze(0)
        return (bg + accumulated).clamp(0, 1)