"""
src/utils/colmap_utils.py
--------------------------
Parse COLMAP TXT output files into Python-friendly structures.

COLMAP produces 3 files after reconstruction:
  - cameras.txt   : Camera intrinsics (focal length, principal point, etc.)
  - images.txt    : Registered images with camera poses (rotation + translation)
  - points3D.txt  : Sparse 3D point cloud

This module parses all three into simple dataclasses.

FIXES APPLIED:
  [FIX-1] ColmapCamera.fy / .cx / .cy properties were wrong for single-focal
          camera models (SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, …).
          Previously relied on len(params) > 2 / ≤ 2 heuristics which pick
          the wrong field index for these models (params[1] is cx, not fy).
          Now uses a model-name → (fi, fj, ci, di) lookup table identical to
          preprocessing/utils._CAMERA_PARAMS. Unknown models fall back to
          OPENCV-style convention (4+ params) or SIMPLE_PINHOLE (3 params).
  [FIX-2] read_images() robust header parser preserved (field-count == 10).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Camera-model intrinsic index table
# (fi, fj, ci, di) = indices into params[] for (fx, fy, cx, cy)
# Single-focal models: fi == fj (one focal length for both axes).
# ---------------------------------------------------------------------------

_CAMERA_PARAM_INDICES: Dict[str, tuple] = {
    # model                    fx   fy   cx   cy
    "SIMPLE_PINHOLE":         (0,   0,   1,   2),
    "PINHOLE":                (0,   1,   2,   3),
    "SIMPLE_RADIAL":          (0,   0,   1,   2),
    "RADIAL":                 (0,   0,   1,   2),
    "OPENCV":                 (0,   1,   2,   3),
    "FULL_OPENCV":            (0,   1,   2,   3),
    "SIMPLE_RADIAL_FISHEYE":  (0,   0,   1,   2),
    "RADIAL_FISHEYE":         (0,   0,   1,   2),
    "OPENCV_FISHEYE":         (0,   1,   2,   3),
    "FOV":                    (0,   1,   2,   3),
    "THIN_PRISM_FISHEYE":     (0,   1,   2,   3),
}


def _intrinsics_from_model(model: str, params: np.ndarray, width: int, height: int):
    """Return (fx, fy, cx, cy) for any COLMAP camera model."""
    indices = _CAMERA_PARAM_INDICES.get(model.upper())
    if indices is None:
        # Unknown model — infer from param count
        if len(params) >= 4:
            indices = (0, 1, 2, 3)
        elif len(params) >= 3:
            indices = (0, 0, 1, 2)
        else:
            f = float(max(width, height))
            return f, f, width / 2.0, height / 2.0
    fi, fj, ci, di = indices
    return float(params[fi]), float(params[fj]), float(params[ci]), float(params[di])


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColmapCamera:
    """One camera model from cameras.txt."""
    camera_id: int
    model:     str          # e.g. "SIMPLE_RADIAL", "OPENCV"
    width:     int
    height:    int
    params:    np.ndarray   # raw parameter array

    # FIX-1: properties now use model-aware index lookup
    @property
    def fx(self) -> float:
        fx, _, _, _ = _intrinsics_from_model(self.model, self.params, self.width, self.height)
        return fx

    @property
    def fy(self) -> float:
        _, fy, _, _ = _intrinsics_from_model(self.model, self.params, self.width, self.height)
        return fy

    @property
    def cx(self) -> float:
        _, _, cx, _ = _intrinsics_from_model(self.model, self.params, self.width, self.height)
        return cx

    @property
    def cy(self) -> float:
        _, _, _, cy = _intrinsics_from_model(self.model, self.params, self.width, self.height)
        return cy


@dataclass
class ColmapImage:
    """One registered image from images.txt."""
    image_id:  int
    qvec:      np.ndarray   # (w, x, y, z) quaternion — world-to-camera rotation
    tvec:      np.ndarray   # (x, y, z) translation   — world-to-camera translation
    camera_id: int
    name:      str

    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3×3 rotation matrix (world→camera)."""
        qw, qx, qy, qz = self.qvec
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

    def world_to_cam(self) -> np.ndarray:
        """Return 4×4 world-to-camera extrinsic matrix."""
        R   = self.rotation_matrix()
        t   = self.tvec.reshape(3, 1)
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3:] = t
        return mat

    def cam_to_world(self) -> np.ndarray:
        """Return 4×4 camera-to-world matrix."""
        return np.linalg.inv(self.world_to_cam())

    def camera_center(self) -> np.ndarray:
        """Return camera centre in world coordinates, shape (3,)."""
        R = self.rotation_matrix()
        return (-R.T @ self.tvec).astype(np.float64)


@dataclass
class ColmapPoint3D:
    """One sparse 3D point from points3D.txt."""
    point_id:  int
    xyz:       np.ndarray       # (3,)  float64
    rgb:       np.ndarray       # (3,)  uint8
    error:     float            # mean reprojection error (px)
    track:     List[tuple]      # [(image_id, point2d_idx), ...]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def read_cameras(path: str) -> Dict[int, ColmapCamera]:
    """
    Parse cameras.txt → {camera_id: ColmapCamera}.

    cameras.txt format:
      # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
      1 SIMPLE_RADIAL 1280 720 800.0 640.0 360.0 0.01
    """
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            width  = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]], dtype=np.float64)
            cameras[cam_id] = ColmapCamera(cam_id, model, width, height, params)
    return cameras


def read_images(path: str) -> Dict[int, ColmapImage]:
    """
    Parse images.txt → {image_id: ColmapImage}.

    images.txt format (2 lines per registered image):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] as (X, Y, POINT3D_ID)   ← may be absent for 0-match images

    FIX-2: Robust parser — detects header lines by field count (exactly 10)
    rather than alternating-line parity, which breaks when COLMAP omits the
    keypoints line for images with zero matches.
    """
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 10:
            try:
                image_id  = int(parts[0])
                float(parts[1])          # QW sanity check
            except ValueError:
                i += 1
                continue

            qvec      = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec      = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name      = parts[9]
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)

            # Skip the optional keypoints line that follows
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split()
                # Next line is a keypoints line if it is NOT an image header
                if not (len(next_parts) == 10 and _is_int(next_parts[0])):
                    i += 1   # consume the keypoints line
        i += 1

    return images


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_points3D(path: str) -> Dict[int, ColmapPoint3D]:
    """
    Parse points3D.txt → {point_id: ColmapPoint3D}.

    points3D.txt format:
      POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split()
            pt_id  = int(parts[0])
            xyz    = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
            rgb    = np.array([int(x)   for x in parts[4:7]], dtype=np.uint8)
            error  = float(parts[7])
            track_raw = [int(x) for x in parts[8:]]
            track     = [(track_raw[j], track_raw[j + 1]) for j in range(0, len(track_raw), 2)]
            points[pt_id] = ColmapPoint3D(pt_id, xyz, rgb, error, track)
    return points


def load_colmap_model(sparse_dir: str) -> tuple:
    """
    Load the full COLMAP text model from *sparse_dir*.

    Accepts any of three equivalent inputs:
      - the ``sparse_text/`` directory directly  (canonical)
      - the ``sparse/`` binary directory          → redirects to sibling ``sparse_text/``
      - the COLMAP ``output_dir/`` parent         → redirects to ``output_dir/sparse_text/``

    Returns (cameras, images, points3D) as dicts keyed by their IDs.
    """
    import logging as _logging
    d = Path(sparse_dir)

    if not (d / "cameras.txt").exists():
        for candidate in [d / "sparse_text", d.parent / "sparse_text"]:
            if (candidate / "cameras.txt").exists():
                _logging.getLogger("monosplat.utils.colmap").warning(
                    "sparse_path %r has no cameras.txt; redirecting to %r",
                    str(d), str(candidate),
                )
                d = candidate
                break

    cameras  = read_cameras(str(d / "cameras.txt"))
    images   = read_images(str(d  / "images.txt"))
    points3D = read_points3D(str(d / "points3D.txt"))

    print(f"✅ COLMAP model loaded:")
    print(f"   Cameras  : {len(cameras)}")
    print(f"   Images   : {len(images)}")
    print(f"   3D Points: {len(points3D):,}")
    return cameras, images, points3D


def get_sparse_point_cloud(points3D: Dict[int, ColmapPoint3D]) -> tuple:
    """
    Extract XYZ positions and normalised RGB colours from the sparse point cloud.

    Returns (xyz, rgb) as float32 arrays of shape (N, 3).
    """
    xyz_list = [p.xyz              for p in points3D.values()]
    rgb_list = [p.rgb / 255.0      for p in points3D.values()]
    xyz = np.array(xyz_list, dtype=np.float32)
    rgb = np.array(rgb_list, dtype=np.float32)
    return xyz, rgb


# Backwards-compatible aliases (used by Colab notebook)
read_cameras_text  = read_cameras
read_images_text   = read_images
read_points3D_text = read_points3D