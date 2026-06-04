"""
camera.py
Viewer Camera for Gaussian Splatting rendering.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from utils.math_utils import look_at, perspective_matrix, quaternion_to_rotation_matrix
from preprocessing.utils import _extract_intrinsics


class Camera:
    """
    Pinhole camera with pre-computed view / projection matrices.

    gsplat-compatible properties:
        FoVx, FoVy     : horizontal / vertical field of view (radians)
        tanfovx, tanfovy: tan(FoV/2)
        image_width, image_height
        world_view_transform : (4,4) float32
        full_proj_transform  : (4,4) float32
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
    # gsplat-compatible properties
    # ------------------------------------------------------------------

    @property
    def image_width(self) -> int:
        return self.width

    @property
    def image_height(self) -> int:
        return self.height

    @property
    def FoVy(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @property
    def FoVx(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def tanfovx(self) -> float:
        return math.tan(self.FoVx / 2.0)

    @property
    def tanfovy(self) -> float:
        return math.tan(self.FoVy / 2.0)

    @property
    def world_view_transform(self) -> np.ndarray:
        return self.view_matrix

    @property
    def full_proj_transform(self) -> np.ndarray:
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

        # Use _extract_intrinsics to handle ALL COLMAP camera models safely
        fx, fy, cx, cy = _extract_intrinsics(
            cam_data.model, cam_data.params, cam_data.width, cam_data.height
        )

        fov_deg = math.degrees(2.0 * math.atan(cam_data.height / (2.0 * max(fy, 1.0))))

        # Scale intrinsics from original COLMAP resolution to viewer resolution
        orig_w = max(cam_data.width,  1)
        orig_h = max(cam_data.height, 1)
        scale  = min(width / orig_w, height / orig_h)

        return cls(
            position    = position,
            width       = width,
            height      = height,
            fov_deg     = fov_deg,
            near        = near,
            far         = far,
            fx          = fx * scale,
            fy          = fy * scale,
            cx          = cx * scale,
            cy          = cy * scale,
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