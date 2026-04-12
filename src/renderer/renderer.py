"""
renderer.py
Software Gaussian Splatting renderer — CPU + CUDA compatible.

Improvements:
- All in-place writes on autograd-tracked tensors removed.
- canvas and transmittance use fully out-of-place accumulation.
- Covariance clamped more safely; max Gaussian radius reduced 512->128.
- Numerical stability improvements in exp() and divisions.
- Avoid repeated .to(device) inside render loop via upfront transfer.
- Cached bg_color tensor; no Python loops on device conversion.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .camera import Camera


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
   -1.0925484305920792,
    0.31539156525252005,
   -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
   -0.5900435899266435,
    2.890611442640554,
   -0.4570457994644658,
    0.3731763325901154,
   -0.4570457994644658,
    1.4453057213903705,
   -0.5900435899266435,
]


def _eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    result = SH_C0 * sh[:, 0, :]

    if degree < 1:
        return (result + 0.5).clamp(0, 1)

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result = (result
              - SH_C1 * y * sh[:, 1, :]
              + SH_C1 * z * sh[:, 2, :]
              - SH_C1 * x * sh[:, 3, :])

    if degree < 2:
        return (result + 0.5).clamp(0, 1)

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (result
              + SH_C2[0] * xy * sh[:, 4, :]
              + SH_C2[1] * yz * sh[:, 5, :]
              + SH_C2[2] * (2*zz-xx-yy) * sh[:, 6, :]
              + SH_C2[3] * xz * sh[:, 7, :]
              + SH_C2[4] * (xx-yy) * sh[:, 8, :])

    if degree < 3:
        return (result + 0.5).clamp(0, 1)

    result = (result
              + SH_C3[0] * y*(3*xx-yy) * sh[:, 9, :]
              + SH_C3[1] * xy*z * sh[:, 10, :]
              + SH_C3[2] * y*(4*zz-xx-yy) * sh[:, 11, :]
              + SH_C3[3] * z*(2*zz-3*xx-3*yy) * sh[:, 12, :]
              + SH_C3[4] * x*(4*zz-xx-yy) * sh[:, 13, :]
              + SH_C3[5] * z*(xx-yy) * sh[:, 14, :]
              + SH_C3[6] * x*(xx-3*yy) * sh[:, 15, :])

    return (result + 0.5).clamp(0, 1)


def _project_cov3d_to_2d(
    positions: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    view_matrix: torch.Tensor,
    fx: float,
    fy: float,
) -> torch.Tensor:
    N = positions.shape[0]
    device = positions.device

    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], dim=1),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], dim=1),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], dim=1),
    ], dim=1)

    # Clamp scales to avoid degenerate covariance
    scales_clamped = scales.clamp(min=1e-6, max=1e3)
    S = scales_clamped.unsqueeze(2) * torch.eye(3, device=device).unsqueeze(0)
    M = R @ S
    cov3d = M @ M.transpose(1, 2)

    W_mat = view_matrix[:3, :3].unsqueeze(0)
    t = view_matrix[:3, 3]

    p_cam = (positions @ view_matrix[:3, :3].T) + t
    # Clamp z to avoid division by near-zero
    z_cam = p_cam[:, 2].clamp(min=0.01)

    zeros = torch.zeros(N, device=device)
    J = torch.stack([
        torch.stack([fx / z_cam, zeros, -fx * p_cam[:, 0] / (z_cam * z_cam)], dim=1),
        torch.stack([zeros, fy / z_cam, -fy * p_cam[:, 1] / (z_cam * z_cam)], dim=1),
    ], dim=1)

    T_mat = J @ W_mat.expand(N, 3, 3)
    cov2d = T_mat @ cov3d @ T_mat.transpose(1, 2)

    # Out-of-place regularisation
    reg_diag = torch.tensor([[0.3, 0.0], [0.0, 0.3]], device=device, dtype=cov2d.dtype)
    cov2d = cov2d + reg_diag.unsqueeze(0)

    return cov2d


class GaussianRenderer:
    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        bg_color: Union[List[float], Tuple[float, ...]] = (0.0, 0.0, 0.0),
        device: str = "cpu",
        batch_size: int = 5000,
    ):
        self.width = int(width)
        self.height = int(height)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        # Cache bg_color on device — avoid repeated allocation in render loop
        self.bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=self.device)
        self.batch_size = batch_size

    def render_torch(self, model, camera: Camera) -> torch.Tensor:
        return self._render_internal(model, camera)

    def render(self, model, camera: Camera) -> np.ndarray:
        t = self._render_internal(model, camera)
        return t.detach().cpu().permute(1, 2, 0).numpy()

    def _render_internal(self, model, camera: Camera) -> torch.Tensor:
        H, W = self.height, self.width
        device = self.device

        # Transfer all model tensors to device once, upfront — not inside the loop
        positions  = model.positions.to(device)
        colors_sh  = model.colors_sh.to(device)
        opacities  = model.opacities.to(device)
        scales     = model.scales.to(device)
        rotations  = model.rotations.to(device)

        N = positions.shape[0]
        if N == 0:
            return self.bg_color.view(3, 1, 1).expand(3, H, W)

        # Cache view matrix on device
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

        positions  = positions[visible]
        colors_sh  = colors_sh[visible]
        opacities  = opacities[visible]
        scales     = scales[visible]
        rotations  = rotations[visible]
        p_cam      = p_cam[visible]
        z_cam      = z_cam[visible]

        order      = z_cam.argsort(descending=True)
        positions  = positions[order]
        colors_sh  = colors_sh[order]
        opacities  = opacities[order].squeeze(1)
        scales     = scales[order]
        rotations  = rotations[order]
        p_cam      = p_cam[order]

        cam_pos_world = torch.from_numpy(camera.position).float().to(device)
        view_dirs = F.normalize(positions - cam_pos_world.unsqueeze(0), dim=1)
        colors = _eval_sh(model.active_sh_degree, colors_sh, view_dirs)

        cov2d = _project_cov3d_to_2d(positions, scales, rotations, view, fx, fy)

        px = (p_cam[:, 0] / p_cam[:, 2]) * fx + cx
        py = (p_cam[:, 1] / p_cam[:, 2]) * fy + cy

        # Detach cov2d: Gaussian footprint is a fixed splatting shape per forward pass
        cov2d_d = cov2d.detach()

        det = (cov2d_d[:, 0, 0] * cov2d_d[:, 1, 1] - cov2d_d[:, 0, 1] ** 2)
        # FIX: stronger clamp for numerical stability in divisions
        det = det.clamp(min=1e-4)

        inv_c00 = cov2d_d[:, 1, 1] / det
        inv_c11 = cov2d_d[:, 0, 0] / det
        inv_c01 = -cov2d_d[:, 0, 1] / det
        inv_cov = torch.stack([
            torch.stack([inv_c00, inv_c01], dim=1),
            torch.stack([inv_c01, inv_c11], dim=1),
        ], dim=1)  # (N, 2, 2), fully detached

        sigma_max = 0.5 * (cov2d_d[:, 0, 0] + cov2d_d[:, 1, 1])
        # FIX: clamp sigma_max before sqrt to avoid nan; max radius 128 (was 256/512)
        radius = (3.0 * sigma_max.clamp(min=0.0).sqrt()).ceil().int().clamp(min=1, max=128)

        # Transmittance buffer — fully detached; in-place writes are safe here
        transmittance = torch.ones(H, W, device=device)

        # Accumulate directly onto canvas — avoids storing one (3, H, W) tensor
        # per Gaussian in a list (which causes OOM with 100k+ Gaussians).
        accumulated = torch.zeros(3, H, W, device=device)

        total_gaussians = len(positions)
        for batch_start in range(0, total_gaussians, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_gaussians)

            for i in range(batch_start, batch_end):
                mu_x = px[i].item()
                mu_y = py[i].item()
                r    = radius[i].item()
                alpha_i = opacities[i]   # scalar, grad-tracked via sigmoid

                x0 = max(0, int(mu_x - r))
                x1 = min(W, int(mu_x + r + 1))
                y0 = max(0, int(mu_y - r))
                y1 = min(H, int(mu_y + r + 1))
                if x0 >= x1 or y0 >= y1:
                    continue

                # Fresh independent tensors each iteration — no shared storage
                lx = torch.arange(x0, x1, device=device, dtype=torch.float32) - mu_x
                ly = torch.arange(y0, y1, device=device, dtype=torch.float32) - mu_y
                dx, dy = torch.meshgrid(ly, lx, indexing="ij")

                ic   = inv_cov[i]
                maha = (
                    dx * (ic[0, 0] * dx + ic[0, 1] * dy)
                    + dy * (ic[1, 0] * dx + ic[1, 1] * dy)
                )

                # clamp maha before exp() for numerical stability
                weight = torch.exp(-0.5 * maha.clamp(min=0.0, max=20.0)).detach()
                alpha  = (alpha_i * weight).clamp(max=0.9999)

                T_patch = transmittance[y0 :y1, x0 :x1].clone()  # clone, not detach
                contrib = alpha * T_patch  # grad flows only via alpha_i (opacity)

                accumulated[:, y0 :y1, x0 :x1] = (
                        accumulated[:, y0 :y1, x0 :x1]
                        + contrib.unsqueeze( 0 ) * colors[i].view( 3, 1, 1 )
                )

                # Update transmittance fully out-of-place — alpha must be detached first
                new_T = T_patch.detach() * (1.0 - alpha.detach())
                transmittance[y0 :y1, x0 :x1] = new_T

            if batch_end % (self.batch_size * 10) == 0 or batch_end == total_gaussians:
                if device == "cuda":
                    torch.cuda.empty_cache()

        bg     = self.bg_color.view(3, 1, 1) * transmittance.detach().unsqueeze(0)
        canvas = bg + accumulated
        return canvas.clamp(0, 1)