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


def perspective_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Return a 4×4 perspective projection matrix (OpenGL convention).

    Args:
        fov_deg: Vertical field of view in degrees.
        aspect:  Width / height ratio.
        near:    Near clipping plane (must be > 0).
        far:     Far clipping plane (must be > near).

    Returns:
        (4, 4) float32 projection matrix.
    """
    if near <= 0 or far <= near:
        raise ValueError(f"Invalid near/far planes: near={near}, far={far}")
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