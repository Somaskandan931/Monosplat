"""
renderer.py
Gaussian Splatting renderer — gsplat-first with software fallback.

Backend priority
----------------
1. gsplat  (nerfstudio-project/gsplat) — actively maintained, CUDA-optimised,
   clean Python API, drop-in replacement for diff-gaussian-rasterization.
   Install:  pip install gsplat

2. Software renderer — pure PyTorch, slow but dependency-free.
   Used automatically on CPU-only machines or when gsplat is not installed.

Why gsplat over the original diff-gaussian-rasterization
---------------------------------------------------------
- Actively maintained (vs. the original repo which has slowed development)
- Cleaner Python API — no separate CUDA extension compile step required
- Supports tile-based rasterization with proper depth sorting
- Exposes rasterization state (radii, depths) needed for densification
- Compatible with the same .splat / .ply export formats
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

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
# gsplat availability check
# ---------------------------------------------------------------------------

def _try_import_gsplat():
    """Try to import gsplat. Return module or None."""
    try:
        import gsplat
        return gsplat
    except ImportError:
        return None


_GSPLAT = _try_import_gsplat()


# ---------------------------------------------------------------------------
# SH evaluation (shared by both paths)
# ---------------------------------------------------------------------------

def _eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics up to given degree."""
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
              + SH_C3[0]*y*(3*xx-yy)*sh[:, 9, :]  + SH_C3[1]*xy*z*sh[:, 10, :]
              + SH_C3[2]*y*(4*zz-xx-yy)*sh[:, 11, :] + SH_C3[3]*z*(2*zz-3*xx-3*yy)*sh[:, 12, :]
              + SH_C3[4]*x*(4*zz-xx-yy)*sh[:, 13, :] + SH_C3[5]*z*(xx-yy)*sh[:, 14, :]
              + SH_C3[6]*x*(xx-3*yy)*sh[:, 15, :])
    return (result + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# 2D covariance projection (software fallback path)
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
    """
    Renders a GaussianModel to an image tensor.

    Uses gsplat (https://github.com/nerfstudio-project/gsplat) when available,
    falls back to a pure-PyTorch software renderer otherwise.
    """

    def __init__(
        self,
        width:    int = 800,
        height:   int = 600,
        bg_color: Union[List[float], Tuple[float, ...]] = (1.0, 1.0, 1.0),
        device:   str = "auto",
        batch_size: int = 5000,
        use_gsplat: bool = True,
    ):
        self.width      = int(width)
        self.height     = int(height)
        self.batch_size = batch_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("[Renderer] ⚠  CUDA requested but not available — falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device

        self.bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=self.device)

        # Use gsplat when available and on CUDA
        self._use_gsplat = (
            use_gsplat
            and _GSPLAT is not None
            and self.device == "cuda"
        )

        if self._use_gsplat:
            print("[Renderer] ✓ Using gsplat CUDA rasterizer (fast path)")
            print(f"           gsplat version: {getattr(_GSPLAT, '__version__', 'unknown')}")
        else:
            if use_gsplat and self.device == "cuda":
                print("[Renderer] ⚠  gsplat not found — using software renderer")
                print("             Install:  pip install gsplat")
            else:
                print(f"[Renderer] Using software renderer ({self.device.upper()})")
                if self.device == "cpu":
                    import warnings
                    warnings.warn(
                        "[Renderer] Running on CPU without gsplat. "
                        "Output quality will be degraded and training will be very slow. "
                        "For production use, run on a CUDA GPU with: pip install gsplat",
                        RuntimeWarning, stacklevel=2,
                    )

    def render_torch(self, model, camera: Camera) -> torch.Tensor:
        """
        Render model from camera viewpoint.
        Returns (3, H, W) float32 tensor in [0, 1].
        """
        if self._use_gsplat:
            return self._render_gsplat(model, camera)

        n = len(model) if hasattr(model, '__len__') else 0
        if n > 5000:
            import warnings
            warnings.warn(
                f"[Renderer] Software renderer called with {n:,} Gaussians — "
                "this will be extremely slow. Install gsplat for GPU acceleration.",
                RuntimeWarning, stacklevel=2,
            )
        return self._render_software(model, camera)

    def render(self, model, camera: Camera) -> np.ndarray:
        """Render and return (H, W, 3) uint8 numpy array."""
        t = self.render_torch(model, camera)
        return (t.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    def render_preview_frames(
        self,
        model,
        cameras: list,
        output_dir: str,
        prefix: str = "preview",
    ) -> list:
        """
        Render a batch of preview frames and save as JPEGs.

        Used for streaming progress previews during training and for
        generating shareable preview grids. Returns list of saved paths.

        Args:
            model:      GaussianModel to render
            cameras:    list of Camera objects
            output_dir: directory to write frames into
            prefix:     filename prefix

        Returns:
            List of absolute paths to saved preview images.
        """
        from pathlib import Path as _Path
        from PIL import Image as _Image

        out = _Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []

        for i, cam in enumerate(cameras):
            try:
                img_np = self.render(model, cam)
                save_path = out / f"{prefix}_{i:04d}.jpg"
                _Image.fromarray(img_np).save(str(save_path), quality=88, optimize=True)
                paths.append(str(save_path))
            except Exception as e:
                import warnings
                warnings.warn(f"[Renderer] Preview frame {i} failed: {e}", RuntimeWarning)

        return paths

    # ------------------------------------------------------------------
    # gsplat path — nerfstudio-project/gsplat
    # ------------------------------------------------------------------

    def _render_gsplat(self, model, camera: Camera) -> torch.Tensor:
        """
        Render using gsplat's rasterization_legacy or rasterization API.

        gsplat >= 1.0 exposes gsplat.rasterization() which handles:
        - Tile-based depth sorting
        - SH evaluation
        - Alpha compositing
        - Proper gradient flow for densification

        https://docs.gsplat.studio/main/apis/rasterization.html
        """
        gs = _GSPLAT

        positions = model.positions.to(self.device)       # (N, 3)
        quats     = model.get_rotation.to(self.device)    # (N, 4) normalised
        scales    = model.get_scaling.to(self.device)     # (N, 3) exp-activated
        opacities = model.get_opacity.to(self.device).squeeze(-1)  # (N,)

        # Camera intrinsics
        fx = torch.tensor(camera.fx, device=self.device, dtype=torch.float32)
        fy = torch.tensor(camera.fy, device=self.device, dtype=torch.float32)
        cx = torch.tensor(camera.cx, device=self.device, dtype=torch.float32)
        cy = torch.tensor(camera.cy, device=self.device, dtype=torch.float32)

        # World-to-camera transform (4×4)
        viewmat = torch.from_numpy(camera.world_view_transform).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 4, 4)

        # Intrinsic matrix K (3×3)
        Ks = torch.zeros(1, 3, 3, device=self.device, dtype=torch.float32)
        Ks[0, 0, 0] = fx
        Ks[0, 1, 1] = fy
        Ks[0, 0, 2] = cx
        Ks[0, 1, 2] = cy
        Ks[0, 2, 2] = 1.0

        # Evaluate colours from SH coefficients for current view direction
        sh_coeffs = model.get_features().to(self.device)  # (N, (deg+1)^2, 3)
        cam_pos   = torch.from_numpy(camera.position).to(self.device, dtype=torch.float32)
        view_dirs = F.normalize(positions - cam_pos.unsqueeze(0), dim=1)
        colors    = _eval_sh(model.active_sh_degree, sh_coeffs, view_dirs)  # (N, 3)

        try:
            # gsplat >= 1.0 unified API
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
                sh_degree= 0,   # SH already evaluated above; pass degree=0
                near_plane  = camera.near,
                far_plane   = camera.far,
                backgrounds = self.bg_color.unsqueeze(0),   # (1, 3)
                packed   = True,   # memory-efficient packed mode
            )
            # render_colors: (1, H, W, 3) → (3, H, W)
            return render_colors[0].permute(2, 0, 1).clamp(0, 1)

        except TypeError:
            # Fallback for older gsplat API (< 1.0) that used rasterize_gaussians
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
                backgrounds = self.bg_color.unsqueeze(0),
            )
            return render_colors[0].permute(2, 0, 1).clamp(0, 1)

    # ------------------------------------------------------------------
    # Software path — pure PyTorch, no external dependency
    # ------------------------------------------------------------------

    def _render_software(self, model, camera: Camera) -> torch.Tensor:
        """
        Software Gaussian rasterizer.
        Correct but slow — suitable only for debugging or CPU demo runs.
        """
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
                ic     = inv_cov[i]
                maha   = dx*(ic[0,0]*dx + ic[0,1]*dy) + dy*(ic[1,0]*dx + ic[1,1]*dy)
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