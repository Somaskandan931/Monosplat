"""
preprocessing/normalize_scene.py
---------------------------------
Scene-scale normalization for MonoSplat.

Translates the scene centroid to the origin and scales it so all
camera centres (and 3-D points) fit inside the unit sphere.  This
prevents Gaussian explosion that occurs when the COLMAP reconstruction
is far from the origin or has wildly mismatched scale.

API
---
normalize_scene(images, points3D) -> (images, points3D, norm_info)
scene_stats(images, points3D)     -> dict with diagnostic values

Both functions accept and return the standard COLMAP dict format
produced by utils.colmap_utils.load_colmap_model().
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_scene(
    images:   Dict,
    points3D: Dict,
) -> Tuple[Dict, Dict, Dict]:
    """
    Translate and scale the scene into the unit sphere.

    Algorithm
    ---------
    1. Compute the centroid of all camera positions.
    2. Subtract the centroid from every camera translation and every 3-D point.
    3. Compute the maximum absolute value across all translated positions.
    4. Divide by that value so everything sits in [-1, 1].

    Returns
    -------
    images_out  : dict — normalized ColmapImage objects (deep-copied)
    points3D_out: dict — normalized ColmapPoint3D objects (deep-copied)
    norm_info   : dict — {"centroid": np.ndarray, "scale": float, "applied": bool}
    """
    images   = copy.deepcopy(images)
    points3D = copy.deepcopy(points3D)

    # ── Step 1: collect camera centres ───────────────────────────────────────
    centers = np.array(
        [img.camera_center() for img in images.values()],
        dtype=np.float64,
    )

    if len(centers) == 0:
        log.warning("[normalize_scene] No camera centres found. Skipping.")
        return images, points3D, {"centroid": np.zeros(3), "scale": 1.0, "applied": False}

    centroid = centers.mean(axis=0)  # (3,)

    # ── Step 2: collect all positions for scale estimation ───────────────────
    point_xyz = np.array(
        [pt.xyz for pt in points3D.values()],
        dtype=np.float64,
    ) if points3D else np.empty((0, 3), dtype=np.float64)

    all_positions = np.concatenate(
        [centers - centroid, point_xyz - centroid] if len(point_xyz) else [centers - centroid],
        axis=0,
    )

    max_val = np.max(np.abs(all_positions))
    if max_val < 1e-9:
        log.warning("[normalize_scene] Scene extent is near-zero. Skipping scale.")
        scale = 1.0
    else:
        scale = 1.0 / max_val

    log.info(
        f"[normalize_scene] centroid={centroid.round(4).tolist()}  "
        f"max_extent={max_val:.4f}  scale={scale:.6f}"
    )

    # ── Step 3: apply to camera extrinsics ───────────────────────────────────
    # ColmapImage stores tvec (world→camera translation).
    # camera_center() = -R^T @ tvec.
    # To shift the world-space origin to `centroid` and scale by `scale`:
    #   new_camera_center = (old_camera_center - centroid) * scale
    # In tvec-space:  tvec = R @ (−camera_center)
    #   new_tvec = R @ (-(old_center - centroid) * scale)
    #            = R @ (-old_center + centroid) * scale
    #            = (R @ (-old_center)) * scale + R @ centroid * scale
    #            = old_tvec * scale + R @ centroid * scale
    # Simplified: new_tvec = (old_tvec + R @ centroid) * scale
    # … but it is cleaner to compute via the centre directly.
    for img in images.values():
        R   = img.rotation_matrix()          # (3,3)
        old_center = img.camera_center()     # (3,)
        new_center = (old_center - centroid) * scale
        img.tvec = (-R @ new_center).astype(np.float64)

    # ── Step 4: apply to 3-D points ──────────────────────────────────────────
    for pt in points3D.values():
        pt.xyz = ((np.asarray(pt.xyz, dtype=np.float64) - centroid) * scale)

    norm_info = {
        "centroid": centroid,
        "scale":    scale,
        "applied":  True,
    }
    return images, points3D, norm_info


def scene_stats(images: Dict, points3D: Dict) -> Dict:
    """
    Return diagnostic statistics about the scene (before or after normalization).

    Keys returned
    -------------
    camera_extent : float  — median pairwise camera distance
    centroid      : list   — mean camera centre
    max_abs_xyz   : float  — max |xyz| across all 3-D points
    n_cameras     : int
    n_points      : int
    """
    centers = np.array(
        [img.camera_center() for img in images.values()],
        dtype=np.float64,
    ) if images else np.empty((0, 3))

    point_xyz = np.array(
        [pt.xyz for pt in points3D.values()],
        dtype=np.float64,
    ) if points3D else np.empty((0, 3))

    n_cam = len(centers)
    if n_cam >= 2:
        diff  = centers[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        upper = dists[np.triu_indices(n_cam, k=1)]
        extent = float(np.median(upper)) if len(upper) > 0 else 0.0
    else:
        extent = 0.0

    return {
        "camera_extent": round(extent, 6),
        "centroid":      centers.mean(axis=0).round(6).tolist() if n_cam > 0 else [0, 0, 0],
        "max_abs_xyz":   float(np.max(np.abs(point_xyz))) if len(point_xyz) > 0 else 0.0,
        "n_cameras":     n_cam,
        "n_points":      len(point_xyz),
    }
