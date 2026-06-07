"""
math_utils.py
3D math helpers: rotation matrices, quaternions, view/projection matrices.

All functions return float32 arrays for GPU-friendliness.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ], dtype=np.float32)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float32)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to 3×3 rotation matrix.

    Args:
        q: array-like of length 4, (w, x, y, z). Need not be unit-length.

    Returns:
        (3, 3) float32 rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Camera / projection helpers
# ---------------------------------------------------------------------------

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Return a 4×4 view matrix (OpenGL / right-handed convention).

    Args:
        eye:    Camera position in world space.
        target: Point the camera looks at.
        up:     World up vector (e.g. [0, 1, 0]).

    Returns:
        (4, 4) float32 view matrix.
    """
    eye    = np.asarray(eye,    dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up     = np.asarray(up,     dtype=np.float64)

    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-8:
        raise ValueError("eye and target are too close together.")
    f /= f_norm

    r = np.cross(f, up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-8:
        raise ValueError("up vector is parallel to the view direction.")
    r /= r_norm

    u = np.cross(r, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3]  = float(-np.dot(r, eye))
    m[1, 3]  = float(-np.dot(u, eye))
    m[2, 3]  = float( np.dot(f, eye))
    return m


def proj_matrix_3dgs(fovx_rad: float, fovy_rad: float, near: float, far: float) -> np.ndarray:
    """
    Return a 4×4 projection matrix matching the original Graphdeco gaussian-splatting
    convention exactly (graphics_utils.py::getProjectionMatrix, z_sign=+1).

    Convention notes
    ----------------
    * P[3,2] = +1.0  → the homogeneous weight w = z_view  (positive-z forward,
      matching COLMAP / OpenCV / diff-gaussian-rasterization).
    * Depth NDC ∈ [0, 1]:  0 at the near plane, 1 at the far plane.
    * fx and fy are treated independently, so non-square-pixel sensors work
      correctly (fx ≠ fy).

    IMPORTANT — transpose before passing to the CUDA rasterizer
    -----------------------------------------------------------
    diff-gaussian-rasterization reads matrices in **column-major** order
    (GLM / Fortran layout), while numpy/PyTorch store arrays in row-major
    (C layout).  The caller (Camera.full_proj_transform) handles the transpose;
    do NOT transpose this matrix directly.

    The original 3DGS code does:
        projection_matrix = getProjectionMatrix(...).transpose(0, 1)   # store P^T
        full_proj = world_view_transform @ projection_matrix           # = W2C^T @ P^T

    This function returns P (un-transposed); Camera.full_proj_transform
    computes the equivalent: view_matrix.T @ proj_matrix.T

    Args:
        fovx_rad: Horizontal field of view in **radians**.
        fovy_rad: Vertical   field of view in **radians**.
        near:     Near clipping plane (must be > 0).
        far:      Far  clipping plane (must be > near).

    Returns:
        (4, 4) float32 projection matrix P  (row-major, un-transposed).
    """
    if near <= 0 or far <= near:
        raise ValueError(f"Invalid near/far planes: near={near}, far={far}")
    import math as _math
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / _math.tan(fovx_rad / 2.0)   # 1/tanHalfFovX  (= fx / (W/2) when centred)
    P[1, 1] = 1.0 / _math.tan(fovy_rad / 2.0)   # 1/tanHalfFovY  (= fy / (H/2) when centred)
    P[3, 2] = 1.0                                 # w = z_view  (perspective divide by depth)
    P[2, 2] = far / (far - near)                  # depth NDC: 0 at near, 1 at far
    P[2, 3] = -(far * near) / (far - near)
    return P


def perspective_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    **DEPRECATED** — kept for backward-compatibility with non-CUDA callers only.

    For the diff-gaussian-rasterization CUDA kernel use proj_matrix_3dgs() instead.
    This function uses the OpenGL convention (P[3,2]=-1, z NDC ∈ [-1,+1]) which is
    INCOMPATIBLE with the 3DGS CUDA rasterizer.

    Args:
        fov_deg: Vertical field of view in degrees.
        aspect:  Width / height ratio.
        near:    Near clipping plane (must be > 0).
        far:     Far clipping plane (must be > near).

    Returns:
        (4, 4) float32 projection matrix (OpenGL convention).
    """
    if near <= 0 or far <= near:
        raise ValueError(f"Invalid near/far planes: near={near}, far={far}")
    import math as _math
    import warnings
    warnings.warn(
        "perspective_matrix() uses OpenGL convention and is incompatible with "
        "diff-gaussian-rasterization.  Use proj_matrix_3dgs() for training/rendering.",
        DeprecationWarning, stacklevel=2,
    )
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


# ---------------------------------------------------------------------------
# Gaussian math
# ---------------------------------------------------------------------------

def build_covariance_3d(scale: np.ndarray, rotation_q: np.ndarray) -> np.ndarray:
    """
    Build 3D covariance matrix Σ = R S Sᵀ Rᵀ from scale and rotation.

    Args:
        scale:      (3,) scale vector (positive, already exp-activated).
        rotation_q: (4,) quaternion (w, x, y, z).

    Returns:
        (3, 3) float32 covariance matrix.
    """
    S = np.diag(np.asarray(scale, dtype=np.float64))
    R = quaternion_to_rotation_matrix(rotation_q).astype(np.float64)
    M = R @ S
    return (M @ M.T).astype(np.float32)


def project_gaussian_2d(
    cov3d: np.ndarray,
    view_matrix: np.ndarray,
    focal_x: float,
    focal_y: float,
    position: np.ndarray,
) -> np.ndarray:
    """
    Project a 3D Gaussian covariance to 2D screen-space (EWA splatting).

    Args:
        cov3d:       (3, 3) 3D covariance.
        view_matrix: (4, 4) camera view matrix.
        focal_x:     Focal length in pixels (x-axis).
        focal_y:     Focal length in pixels (y-axis).
        position:    (3,) Gaussian center in world space.

    Returns:
        (2, 2) float32 2D covariance matrix.

    Raises:
        ValueError: If the Gaussian is behind the camera (z ≤ 0).
    """
    W     = view_matrix[:3, :3].astype(np.float64)
    t     = view_matrix[:3,  3].astype(np.float64)
    p_cam = W @ np.asarray(position, dtype=np.float64) + t
    z     = p_cam[2]
    if z <= 0:
        raise ValueError(f"Gaussian center is behind the camera (z={z:.4f}).")

    z2  = z * z
    J   = np.array([
        [focal_x / z, 0.0,         -focal_x * p_cam[0] / z2],
        [0.0,         focal_y / z, -focal_y * p_cam[1] / z2],
    ], dtype=np.float64)

    cov3d_d = cov3d.astype(np.float64)
    cov2d   = J @ W @ cov3d_d @ W.T @ J.T
    return cov2d.astype(np.float32)