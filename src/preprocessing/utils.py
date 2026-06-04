"""
preprocessing/utils.py
Helpers for reading COLMAP text-format outputs into Python structures.

Supports:
    cameras.txt   → dict[int, Camera]
    images.txt    → dict[int, Image]
    points3D.txt  → dict[int, Point3D]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    camera_id: int
    model:     str
    width:     int
    height:    int
    params:    np.ndarray   # fx, fy, cx, cy, [k1, k2, p1, p2, ...]


@dataclass
class Image:
    image_id:     int
    qvec:         np.ndarray   # (w, x, y, z) quaternion
    tvec:         np.ndarray   # (x, y, z) translation
    camera_id:    int
    name:         str
    xys:          np.ndarray   # (N, 2) 2D keypoint positions
    point3D_ids:  np.ndarray   # (N,)   matching 3D point IDs (−1 = unmatched)


@dataclass
class Point3D:
    point3D_id:   int
    xyz:          np.ndarray   # (3,) float64
    rgb:          np.ndarray   # (3,) uint8
    error:        float
    image_ids:    np.ndarray   # (M,) int64
    point2D_idxs: np.ndarray   # (M,) int64


# ---------------------------------------------------------------------------
# Camera model param extraction (fixes 'NoneType' has no attribute CameraModelType)
# ---------------------------------------------------------------------------

_CAMERA_PARAMS = {
    # model_name -> (fx_idx, fy_idx, cx_idx, cy_idx)
    # SIMPLE_PINHOLE: f, cx, cy
    "SIMPLE_PINHOLE": (0, 0, 1, 2),
    # PINHOLE: fx, fy, cx, cy
    "PINHOLE":        (0, 1, 2, 3),
    # SIMPLE_RADIAL: f, cx, cy, k1
    "SIMPLE_RADIAL":  (0, 0, 1, 2),
    # RADIAL: f, cx, cy, k1, k2
    "RADIAL":         (0, 0, 1, 2),
    # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV":         (0, 1, 2, 3),
    # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    "FULL_OPENCV":    (0, 1, 2, 3),
    # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
    "SIMPLE_RADIAL_FISHEYE": (0, 0, 1, 2),
    # RADIAL_FISHEYE: f, cx, cy, k1, k2
    "RADIAL_FISHEYE": (0, 0, 1, 2),
    # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
    "OPENCV_FISHEYE": (0, 1, 2, 3),
    # FOV: fx, fy, cx, cy, omega
    "FOV":            (0, 1, 2, 3),
    # THIN_PRISM_FISHEYE: fx, fy, cx, cy, ...
    "THIN_PRISM_FISHEYE": (0, 1, 2, 3),
}


def _extract_intrinsics(model: str, params: np.ndarray, width: int, height: int):
    """Extract (fx, fy, cx, cy) from a COLMAP camera params array for any model."""
    indices = _CAMERA_PARAMS.get(model.upper())
    if indices is None:
        # Unknown model — fall back to OpenCV convention or reasonable defaults
        if len(params) >= 4:
            indices = (0, 1, 2, 3)
        elif len(params) >= 3:
            indices = (0, 0, 1, 2)
        else:
            # Absolute fallback: assume square pixels, principal point at center
            f = max(width, height)
            return float(f), float(f), width / 2.0, height / 2.0

    fi, fj, ci, di = indices
    fx = float(params[fi])
    fy = float(params[fj])
    cx = float(params[ci])
    cy = float(params[di])
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_cameras(path) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = Camera(
                camera_id=cam_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=np.array(list(map(float, parts[4:])), dtype=np.float64),
            )
    return cameras


def read_images(path) -> Dict[int, Image]:
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while i < len(lines):
        parts  = lines[i].split()
        # Image header: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (10 fields)
        if len(parts) != 10:
            i += 1
            continue
        try:
            img_id = int(parts[0])
            float(parts[1])  # QW sanity check
        except ValueError:
            i += 1
            continue

        qvec   = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec   = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        name   = parts[9]

        # Second line: 2D keypoints (may be empty for images with no matches)
        if i + 1 < len(lines) and len(lines[i + 1].split()) % 3 == 0 and lines[i + 1].split() and _is_keypoint_line(lines[i + 1]):
            pts_parts = lines[i + 1].split()
            n    = len(pts_parts) // 3
            xys  = np.array(
                [[float(pts_parts[j * 3]), float(pts_parts[j * 3 + 1])] for j in range(n)],
                dtype=np.float64,
            )
            pt3d_ids = np.array([int(pts_parts[j * 3 + 2]) for j in range(n)], dtype=np.int64)
            i += 2
        else:
            xys      = np.zeros((0, 2), dtype=np.float64)
            pt3d_ids = np.zeros(0, dtype=np.int64)
            i += 1

        images[img_id] = Image(img_id, qvec, tvec, cam_id, name, xys, pt3d_ids)
    return images


def _is_keypoint_line(line: str) -> bool:
    """Return True if a line looks like a COLMAP keypoint line (not an image header)."""
    parts = line.split()
    if not parts:
        return True  # empty keypoint line is valid
    try:
        int(parts[0])  # image header starts with integer IMAGE_ID
        float(parts[1])  # followed by QW float
        # If both succeed AND we have exactly 10 parts, it's a header, not keypoints
        if len(parts) == 10:
            return False
    except (ValueError, IndexError):
        pass
    return True


def read_points3d(path) -> Dict[int, Point3D]:
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split()
            pt_id  = int(parts[0])
            xyz    = np.array(list(map(float, parts[1:4])), dtype=np.float64)
            rgb    = np.array(list(map(int,   parts[4:7])), dtype=np.uint8)
            error  = float(parts[7])
            track  = list(map(int, parts[8:]))
            img_ids   = np.array(track[0::2], dtype=np.int64)
            pt2d_idxs = np.array(track[1::2], dtype=np.int64)
            points[pt_id] = Point3D(pt_id, xyz, rgb, error, img_ids, pt2d_idxs)
    return points


def load_colmap_model(model_dir) -> Tuple[Dict, Dict, Dict]:
    """
    Load a COLMAP text-format sparse model from *model_dir*.

    Returns:
        (cameras, images, points3d) dicts keyed by their respective IDs.
    """
    d = Path(model_dir)
    required = ["cameras.txt", "images.txt", "points3D.txt"]
    for fname in required:
        if not (d / fname).exists():
            raise FileNotFoundError(
                f"Missing COLMAP file: {d / fname}\n"
                "Run COLMAP first: python scripts/prepare_dataset.py"
            )

    cameras  = read_cameras(d / "cameras.txt")
    images   = read_images(d  / "images.txt")
    points3d = read_points3d(d / "points3D.txt")

    print(
        f"[colmap] Loaded model: {len(cameras)} cameras, "
        f"{len(images)} images, {len(points3d):,} 3D points"
    )
    return cameras, images, points3d
