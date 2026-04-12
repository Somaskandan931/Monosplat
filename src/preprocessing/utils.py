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
    while i + 1 < len(lines):
        parts  = lines[i].split()
        img_id = int(parts[0])
        qvec   = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec   = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        name   = parts[9]

        # Second line: 2D keypoints interleaved with 3D point IDs
        pts_parts = lines[i + 1].split()
        n    = len(pts_parts) // 3
        xys  = np.array(
            [[float(pts_parts[j * 3]), float(pts_parts[j * 3 + 1])] for j in range(n)],
            dtype=np.float64,
        )
        pt3d_ids = np.array([int(pts_parts[j * 3 + 2]) for j in range(n)], dtype=np.int64)

        images[img_id] = Image(img_id, qvec, tvec, cam_id, name, xys, pt3d_ids)
        i += 2
    return images


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

    Expected files: cameras.txt, images.txt, points3D.txt

    Returns:
        (cameras, images, points3d) dicts keyed by their respective IDs.

    Raises:
        FileNotFoundError: If any of the three required files is missing.
    """
    d = Path(model_dir)
    required = ["cameras.txt", "images.txt", "points3D.txt"]
    for fname in required:
        if not (d / fname).exists():
            raise FileNotFoundError(
                f"Missing COLMAP file: {d / fname}\n"
                "Run COLMAP first: python -m src.preprocessing.colmap_runner"
            )

    cameras  = read_cameras(d / "cameras.txt")
    images   = read_images(d  / "images.txt")
    points3d = read_points3d(d / "points3D.txt")

    print(
        f"[colmap] Loaded model: {len(cameras)} cameras, "
        f"{len(images)} images, {len(points3d):,} 3D points"
    )
    return cameras, images, points3d