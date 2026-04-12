"""
camera.py
Viewer Camera for Gaussian Splatting rendering.

Holds all intrinsics + extrinsics needed to render a GaussianModel.

Two construction paths
----------------------
    Camera(position, target, up, fov_deg, width, height)
        — direct construction (e.g. for thumbnails)

    Camera.from_colmap(img_data, cam_data, width, height)
        — convert from COLMAP sparse-model Image + Camera records
          (used by pipeline_manager.py and worker.py during training)

The object is intentionally plain-data (no PyTorch tensors) so it can be
passed across threads and pickled safely.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.utils.math_utils import look_at, perspective_matrix, quaternion_to_rotation_matrix


class Camera:
    """
    Pinhole camera with pre-computed view / projection matrices.

    Attributes
    ----------
    width, height   : int        render resolution (pixels)
    position        : (3,) f32   world-space eye position
    target          : (3,) f32   world-space look-at point
    up              : (3,) f32   world-space up vector
    fov_deg         : float      vertical field of view (degrees)
    near, far       : float      clipping planes
    fx, fy          : float      focal lengths in pixels
    cx, cy          : float      principal point in pixels
    view_matrix     : (4,4) f32  world→camera transform
    proj_matrix     : (4,4) f32  camera→clip transform
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
        # Optional override: supply intrinsics directly instead of fov
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        # Optional override: supply full 4×4 view matrix directly
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

        # ── View matrix ──────────────────────────────────────────────
        if view_matrix is not None:
            self.view_matrix = np.asarray(view_matrix, dtype=np.float32)
        else:
            self.view_matrix = look_at(self.position, self.target, self.up)

        # ── Intrinsics ───────────────────────────────────────────────
        # If focal lengths provided directly, use them; else derive from fov
        if fx is not None:
            self.fx = float(fx)
            self.fy = float(fy) if fy is not None else float(fx)
        else:
            # Standard pinhole: fx = (W/2) / tan(fov_h/2)
            # We're given vertical fov, so:
            aspect   = self.width / max(self.height, 1)
            fov_v_r  = math.radians(fov_deg)
            self.fy  = (self.height / 2.0) / math.tan(fov_v_r / 2.0)
            self.fx  = self.fy  # square pixels

        self.cx = float(cx) if cx is not None else self.width  / 2.0
        self.cy = float(cy) if cy is not None else self.height / 2.0

        # ── Projection matrix (OpenGL convention) ────────────────────
        aspect           = self.width / max(self.height, 1)
        self.proj_matrix = perspective_matrix(fov_deg, aspect, near, far)

    # ------------------------------------------------------------------
    # Factory: COLMAP → Camera
    # ------------------------------------------------------------------

    @classmethod
    def from_colmap(
        cls,
        img_data,      # preprocessing.utils.Image dataclass
        cam_data,      # preprocessing.utils.Camera dataclass
        width:  int,
        height: int,
        near:   float = 0.01,
        far:    float = 100.0,
    ) -> "Camera":
        """
        Build a Camera from COLMAP sparse-model records.

        COLMAP stores the world→camera rotation as a quaternion (w,x,y,z)
        and a translation vector such that:
            p_cam = R @ p_world + t

        We need the camera position in world space:
            position = -Rᵀ @ t

        Supported COLMAP camera models: SIMPLE_PINHOLE, PINHOLE, OPENCV,
        SIMPLE_RADIAL, RADIAL.  For all models params[0:4] = (fx, fy, cx, cy)
        or (f, cx, cy) for SIMPLE_PINHOLE/SIMPLE_RADIAL.
        """
        # ── Rotation matrix from quaternion (w,x,y,z) ────────────────
        R = quaternion_to_rotation_matrix(img_data.qvec).astype(np.float64)  # (3,3)
        t = np.asarray(img_data.tvec, dtype=np.float64)                       # (3,)

        # Camera centre in world coords
        position = (-R.T @ t).astype(np.float32)

        # Build 4×4 view matrix directly from R and t
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = R.astype(np.float32)
        view[:3,  3] = t.astype(np.float32)

        # ── Intrinsics from COLMAP camera model ───────────────────────
        params = cam_data.params  # numpy array
        model  = cam_data.model.upper()

        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            # params: f, cx, cy, [k]
            fx = fy = float(params[0])
            cx = float(params[1])
            cy = float(params[2])
        elif model in ("PINHOLE", "OPENCV", "RADIAL", "FULL_OPENCV",
                       "OPENCV_FISHEYE", "FOV"):
            # params: fx, fy, cx, cy, [distortion…]
            fx = float(params[0])
            fy = float(params[1])
            cx = float(params[2])
            cy = float(params[3])
        else:
            # Unknown model — fall back to fx=fy from image width
            fx = fy = float(cam_data.width)
            cx = cam_data.width  / 2.0
            cy = cam_data.height / 2.0

        # Derive fov from fy so perspective_matrix stays consistent
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
            f"{self.width}×{self.height}, fov={self.fov_deg:.1f}°)"
        )