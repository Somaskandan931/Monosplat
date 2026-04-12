"""
renderer.py
Gaussian Splatting renderer — CUDA-first with software fallback.

360GS uses diff-gaussian-rasterization-panorama (a CUDA extension).
This renderer tries the CUDA path first, then falls back to the software
renderer so your code runs on CPU/free-tier Colab without modification.

CUDA path (fast — used in production + Colab T4):
    pip install submodules/diff-gaussian-rasterization
    (or the panorama variant from 360GS for indoor scenes)

Software path (slow — for debugging, CPU-only environments):
    No installation required. Same output, ~100x slower.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .camera import Camera


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.4453057213903705,
          -0.5900435899266435]


# ---------------------------------------------------------------------------
# CUDA rasterizer availability check
# ---------------------------------------------------------------------------

def _try_import_cuda_rasterizer():
    """Try to import diff-gaussian-rasterization. Return module or None."""
    try:
        import diff_gaussian_rasterization as dgr
        return dgr
    except ImportError:
        return None


_CUDA_RASTERIZER = _try_import_cuda_rasterizer()


# ---------------------------------------------------------------------------
# SH evaluation (shared by both paths)
# ---------------------------------------------------------------------------

def _eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    result = SH_C0 * sh[:, 0, :]
    if degree < 1:
        return (result + 0.5).clamp(0, 1)
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result = result - SH_C1*y*sh[:, 1, :] + SH_C1*z*sh[:, 2, :] - SH_C1*x*sh[:, 3, :]
    if degree < 2:
        return (result + 0.5).clamp(0, 1)
    xx, yy, zz = x*x, y*y, z*z
    xy, yz, xz = x*y, y*z, x*z
    result = (result
              + SH_C2[0]*xy*sh[:, 4, :] + SH_C2[1]*yz*sh[:, 5, :]
              + SH_C2[2]*(2*zz-xx-yy)*sh[:, 6, :] + SH_C2[3]*xz*sh[:, 7, :]
              + SH_C2[4]*(xx-yy)*sh[:, 8, :])
    if degree < 3:
        return (result + 0.5).clamp(0, 1)
    result = (result
              + SH_C3[0]*y*(3*xx-yy)*sh[:, 9, :] + SH_C3[1]*xy*z*sh[:, 10, :]
              + SH_C3[2]*y*(4*zz-xx-yy)*sh[:, 11, :] + SH_C3[3]*z*(2*zz-3*xx-3*yy)*sh[:, 12, :]
              + SH_C3[4]*x*(4*zz-xx-yy)*sh[:, 13, :] + SH_C3[5]*z*(xx-yy)*sh[:, 14, :]
              + SH_C3[6]*x*(xx-3*yy)*sh[:, 15, :])
    return (result + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# 2D covariance projection (software path)
# ---------------------------------------------------------------------------

def _project_cov3d_to_2d(
    positions, scales, rotations, view_matrix, fx, fy
) -> torch.Tensor:
    N = positions.shape[0]
    device = positions.device
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=1),
    ], dim=1)
    S      = scales.clamp(1e-6, 1e3).unsqueeze(2) * torch.eye(3, device=device).unsqueeze(0)
    M      = R @ S
    cov3d  = M @ M.transpose(1, 2)
    W_mat  = view_matrix[:3, :3].unsqueeze(0)
    t      = view_matrix[:3, 3]
    p_cam  = (positions @ view_matrix[:3, :3].T) + t
    z_cam  = p_cam[:, 2].clamp(min=0.01)
    zeros  = torch.zeros(N, device=device)
    J = torch.stack([
        torch.stack([fx / z_cam, zeros, -fx * p_cam[:, 0] / (z_cam * z_cam)], dim=1),
        torch.stack([zeros, fy / z_cam, -fy * p_cam[:, 1] / (z_cam * z_cam)], dim=1),
    ], dim=1)
    T_mat  = J @ W_mat.expand(N, 3, 3)
    cov2d  = T_mat @ cov3d @ T_mat.transpose(1, 2)
    reg    = torch.tensor([[0.3, 0.0], [0.0, 0.3]], device=device, dtype=cov2d.dtype)
    return cov2d + reg.unsqueeze(0)


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class GaussianRenderer:
    def __init__(
        self,
        width:    int = 800,
        height:   int = 600,
        bg_color: Union[List[float], Tuple[float, ...]] = (1.0, 1.0, 1.0),
        device:   str = "cpu",
        batch_size: int = 5000,
        use_cuda_rasterizer: bool = True,
    ):
        self.width    = int(width)
        self.height   = int(height)
        self.device   = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=self.device)
        self.batch_size = batch_size

        # Use CUDA rasterizer if available and requested
        self._use_cuda = (
            use_cuda_rasterizer
            and _CUDA_RASTERIZER is not None
            and self.device == "cuda"
        )
        if self._use_cuda:
            print("[Renderer] ✓ Using CUDA diff-gaussian-rasterization (fast path)")
        else:
            if use_cuda_rasterizer and self.device == "cuda":
                print("[Renderer] ⚠  diff-gaussian-rasterization not found — using software renderer")
                print("             Install: pip install submodules/diff-gaussian-rasterization")
            else:
                print("[Renderer] Using software renderer (CPU-compatible)")

    def render_torch(self, model, camera: Camera) -> torch.Tensor:
        if self._use_cuda:
            return self._render_cuda(model, camera)
        return self._render_software(model, camera)

    def render(self, model, camera: Camera) -> np.ndarray:
        t = self.render_torch(model, camera)
        return t.detach().cpu().permute(1, 2, 0).numpy()

    # ------------------------------------------------------------------
    # CUDA path — matches 360GS gaussian_renderer/__init__.py
    # ------------------------------------------------------------------

    def _render_cuda(self, model, camera: Camera) -> torch.Tensor:
        """
        Render using diff-gaussian-rasterization CUDA extension.
        This path is 10–100x faster than the software renderer.
        API matches 360GS gaussian_renderer/__init__.py render() call.
        """
        dgr = _CUDA_RASTERIZER

        positions  = model.positions.to(self.device)
        colors_sh  = model.get_features().to(self.device)
        opacities  = model.get_opacity.to(self.device)
        scales     = model.get_scaling.to(self.device)
        rotations  = model.get_rotation.to(self.device)

        # Rasterization settings (matches 360GS)
        raster_settings = dgr.GaussianRasterizationSettings(
            image_height   = camera.image_height,
            image_width    = camera.image_width,
            tanfovx        = camera.tanfovx,
            tanfovy        = camera.tanfovy,
            bg             = self.bg_color,
            scale_modifier = 1.0,
            viewmatrix     = torch.from_numpy(camera.world_view_transform).to(self.device),
            projmatrix     = torch.from_numpy(camera.full_proj_transform).to(self.device),
            sh_degree      = model.active_sh_degree,
            campos         = torch.from_numpy(camera.position).to(self.device),
            prefiltered    = False,
            debug          = False,
        )

        rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(positions, requires_grad=True)
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        rendered_image, radii = rasterizer(
            means3D        = positions,
            means2D        = screenspace_points,
            shs            = colors_sh,
            colors_precomp = None,
            opacities      = opacities,
            scales         = scales,
            rotations      = rotations,
            cov3D_precomp  = None,
        )
        return rendered_image

    # ------------------------------------------------------------------
    # Software path (unchanged from MonoSplat v1)
    # ------------------------------------------------------------------

    def _render_software(self, model, camera: Camera) -> torch.Tensor:
        H, W   = self.height, self.width
        device = self.device

        positions = model.positions.to(device)
        colors_sh = model.colors_sh.to(device)
        opacities = model.opacities.to(device)
        scales    = model.scales.to(device)
        rotations = model.rotations.to(device)

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

        positions = positions[visible]
        colors_sh = colors_sh[visible]
        opacities = opacities[visible]
        scales    = scales[visible]
        rotations = rotations[visible]
        p_cam     = p_cam[visible]
        z_cam     = z_cam[visible]

        order     = z_cam.argsort(descending=True)
        positions = positions[order];  colors_sh = colors_sh[order]
        opacities = opacities[order].squeeze(1)
        scales    = scales[order];     rotations = rotations[order]
        p_cam     = p_cam[order]

        cam_pos   = torch.from_numpy(camera.position).float().to(device)
        view_dirs = F.normalize(positions - cam_pos.unsqueeze(0), dim=1)
        colors    = _eval_sh(model.active_sh_degree, colors_sh, view_dirs)

        cov2d = _project_cov3d_to_2d(positions, scales, rotations, view, fx, fy)

        px = (p_cam[:, 0] / p_cam[:, 2]) * fx + cx
        py = (p_cam[:, 1] / p_cam[:, 2]) * fy + cy

        cov2d_d = cov2d.detach()
        det     = (cov2d_d[:, 0, 0]*cov2d_d[:, 1, 1] - cov2d_d[:, 0, 1]**2).clamp(min=1e-4)
        inv_c00 = cov2d_d[:, 1, 1] / det
        inv_c11 = cov2d_d[:, 0, 0] / det
        inv_c01 = -cov2d_d[:, 0, 1] / det
        inv_cov = torch.stack([
            torch.stack([inv_c00, inv_c01], dim=1),
            torch.stack([inv_c01, inv_c11], dim=1),
        ], dim=1)

        sigma_max = 0.5 * (cov2d_d[:, 0, 0] + cov2d_d[:, 1, 1])
        radius    = (3.0 * sigma_max.clamp(min=0.0).sqrt()).ceil().int().clamp(min=1, max=128)

        transmittance = torch.ones(H, W, device=device)
        accumulated   = torch.zeros(3, H, W, device=device)

        for batch_start in range(0, len(positions), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(positions))
            for i in range(batch_start, batch_end):
                mu_x, mu_y = px[i].item(), py[i].item()
                r           = radius[i].item()
                alpha_i     = opacities[i]
                x0 = max(0, int(mu_x - r)); x1 = min(W, int(mu_x + r + 1))
                y0 = max(0, int(mu_y - r)); y1 = min(H, int(mu_y + r + 1))
                if x0 >= x1 or y0 >= y1:
                    continue
                lx = torch.arange(x0, x1, device=device, dtype=torch.float32) - mu_x
                ly = torch.arange(y0, y1, device=device, dtype=torch.float32) - mu_y
                dx, dy = torch.meshgrid(ly, lx, indexing="ij")
                ic   = inv_cov[i]
                maha = dx*(ic[0,0]*dx + ic[0,1]*dy) + dy*(ic[1,0]*dx + ic[1,1]*dy)
                weight = torch.exp(-0.5 * maha.clamp(0.0, 20.0)).detach()
                alpha  = (alpha_i * weight).clamp(max=0.9999)
                T_patch = transmittance[y0:y1, x0:x1].clone()
                contrib = alpha * T_patch
                accumulated[:, y0:y1, x0:x1] = (
                    accumulated[:, y0:y1, x0:x1]
                    + contrib.unsqueeze(0) * colors[i].view(3, 1, 1)
                )
                transmittance[y0:y1, x0:x1] = T_patch.detach() * (1.0 - alpha.detach())
            if device == "cuda" and batch_end % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()

        bg     = self.bg_color.view(3, 1, 1) * transmittance.detach().unsqueeze(0)
        canvas = bg + accumulated
        return canvas.clamp(0, 1)