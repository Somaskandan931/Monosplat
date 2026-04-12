"""
camera.py
Viewer Camera for Gaussian Splatting rendering.

Changes from MonoSplat v1 to match LeoDarcy/360GS camera convention
--------------------------------------------------------------------
- Added FoVx / FoVy properties (used by 360GS gaussian_renderer/__init__.py)
- Added tanfovx / tanfovy (used by CUDA rasterizer)
- Added world_view_transform / full_proj_transform (360GS renderer convention)
- Camera is still plain-data (no PyTorch tensors) for thread-safety
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.utils.math_utils import look_at, perspective_matrix, quaternion_to_rotation_matrix


class Camera:
    """
    Pinhole camera with pre-computed view / projection matrices.

    360GS-compatible properties
    ---------------------------
    FoVx, FoVy     : horizontal / vertical field of view (radians)
    tanfovx, tanfovy: tan(FoV/2) — used by CUDA rasterizer
    image_width, image_height : aliases for width/height (360GS style)
    world_view_transform : (4,4) float32 — same as view_matrix
    full_proj_transform  : (4,4) float32 — view @ proj
    """

    def __init__(
        self,
        position:  np.ndarray,
        width:     int,
        height:    int,
        target:    Optional[np.ndarray] = None,
        up:        Optional[np.ndarray] = None,
        fov_deg:   float = 60.0,
        near:      float = 0.01,
        far:       float = 100.0,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        view_matrix: Optional[np.ndarray] = None,
    ):
        self.width    = int(width)
        self.height   = int(height)
        self.fov_deg  = float(fov_deg)
        self.near     = float(near)
        self.far      = float(far)

        self.position = np.asarray(position, dtype=np.float32).flatten()[:3]

        if target is None:
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32).flatten()[:3]
        self.up     = np.asarray(up,     dtype=np.float32).flatten()[:3]

        if view_matrix is not None:
            self.view_matrix = np.asarray(view_matrix, dtype=np.float32)
        else:
            self.view_matrix = look_at(self.position, self.target, self.up)

        if fx is not None:
            self.fx = float(fx)
            self.fy = float(fy) if fy is not None else float(fx)
        else:
            fov_v_r = math.radians(fov_deg)
            self.fy = (self.height / 2.0) / math.tan(fov_v_r / 2.0)
            self.fx = self.fy

        self.cx = float(cx) if cx is not None else self.width  / 2.0
        self.cy = float(cy) if cy is not None else self.height / 2.0

        aspect           = self.width / max(self.height, 1)
        self.proj_matrix = perspective_matrix(fov_deg, aspect, near, far)

    # ------------------------------------------------------------------
    # 360GS-compatible properties
    # ------------------------------------------------------------------

    @property
    def image_width(self) -> int:
        """360GS alias for width."""
        return self.width

    @property
    def image_height(self) -> int:
        """360GS alias for height."""
        return self.height

    @property
    def FoVy(self) -> float:
        """Vertical field of view in radians (360GS convention)."""
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @property
    def FoVx(self) -> float:
        """Horizontal field of view in radians (360GS convention)."""
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def tanfovx(self) -> float:
        """tan(FoVx/2) — used by CUDA diff-gaussian-rasterization."""
        return math.tan(self.FoVx / 2.0)

    @property
    def tanfovy(self) -> float:
        """tan(FoVy/2) — used by CUDA diff-gaussian-rasterization."""
        return math.tan(self.FoVy / 2.0)

    @property
    def world_view_transform(self) -> np.ndarray:
        """360GS alias for view_matrix (world → camera)."""
        return self.view_matrix

    @property
    def full_proj_transform(self) -> np.ndarray:
        """360GS full_proj_transform = view @ proj."""
        return self.view_matrix @ self.proj_matrix

    # ------------------------------------------------------------------
    # Factory: COLMAP → Camera
    # ------------------------------------------------------------------

    @classmethod
    def from_colmap(
        cls,
        img_data,
        cam_data,
        width:  int,
        height: int,
        near:   float = 0.01,
        far:    float = 100.0,
    ) -> "Camera":
        R = quaternion_to_rotation_matrix(img_data.qvec).astype(np.float64)
        t = np.asarray(img_data.tvec, dtype=np.float64)
        position = (-R.T @ t).astype(np.float32)

        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R.astype(np.float32)
        view[:3,  3] = t.astype(np.float32)

        params = cam_data.params
        model  = cam_data.model.upper()

        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = float(params[0])
            cx = float(params[1])
            cy = float(params[2])
        elif model in ("PINHOLE", "OPENCV", "RADIAL", "FULL_OPENCV",
                       "OPENCV_FISHEYE", "FOV"):
            fx = float(params[0])
            fy = float(params[1])
            cx = float(params[2])
            cy = float(params[3])
        else:
            fx = fy = float(cam_data.width)
            cx = cam_data.width  / 2.0
            cy = cam_data.height / 2.0

        fov_deg = math.degrees(2.0 * math.atan(cam_data.height / (2.0 * fy)))

        return cls(
            position    = position,
            width       = width,
            height      = height,
            fov_deg     = fov_deg,
            near        = near,
            far         = far,
            fx          = fx * (width  / max(cam_data.width,  1)),
            fy          = fy * (height / max(cam_data.height, 1)),
            cx          = cx * (width  / max(cam_data.width,  1)),
            cy          = cy * (height / max(cam_data.height, 1)),
            view_matrix = view,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    def __repr__(self) -> str:
        pos = self.position.tolist()
        return (
            f"Camera(pos={[round(v,3) for v in pos]}, "
            f"{self.width}×{self.height}, "
            f"FoVx={math.degrees(self.FoVx):.1f}°, FoVy={math.degrees(self.FoVy):.1f}°)"
        )