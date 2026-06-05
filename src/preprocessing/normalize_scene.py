"""
preprocessing/normalize_scene.py
---------------------------------
Scene-scale normalization for MonoSplat.

ROOT-CAUSE FIX (foggy preview):
  The old implementation used max_abs across ALL 3-D points to compute scale:

      max_val = np.max(np.abs(all_positions))   # pulled in by COLMAP outliers
      scale   = 1.0 / max_val

  With 83 719 points whose P99 distance from centroid is 61 m but max distance
  is 113 m, max_val = 108 and scale = 0.009256.  Camera centres that were
  already at ~5.18 m radius got compressed to 0.048 m radius after scaling.
  train.py then floored cameras_extent to 0.1 (hardcoded guard), meaning:
    - densify_and_prune() pruned nearly every Gaussian (screen-size threshold
      derived from a 10× too small extent)
    - initialise_from_pcd clamped log-scales to max=0 → exp(0)=1 world-unit,
      but the scene was only 0.048 units wide → every Gaussian covered 2100%
      of the scene diameter → solid grey fog from iteration 1.

FIX-1 (scale): Use camera positions only (not point cloud) to compute scale.
  The maximum camera radius is by definition the right unit for normalizing
  a scene so cameras fit inside the unit sphere.  Outlier 3-D points don't
  affect it.

FIX-2 (outliers): Filter point cloud at P99 distance from centroid before
  applying normalization.  This removes COLMAP reconstruction artifacts without
  losing valid scene geometry.

API
---
normalize_scene(images, points3D) -> (images, points3D, norm_info)
scene_stats(images, points3D)     -> dict with diagnostic values

Both functions accept and return the standard COLMAP dict format produced by
utils.colmap_utils.load_colmap_model().
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
    Translate and scale the scene so cameras fit inside the unit sphere.

    Algorithm (FIX-1)
    -----------------
    1. Compute the mean centroid of all camera positions.
    2. Compute the camera radius = max distance from any camera to the centroid.
    3. Scale = 1 / camera_radius  (cameras fill exactly the unit sphere).
    4. Apply the same translation + scale to all 3-D points (after filtering).

    This is strictly better than the old max_abs approach because:
      - It is unaffected by COLMAP point-cloud outliers.
      - It guarantees cameras_extent ≈ 1.0 after normalization.
      - It is mathematically the correct unit for a splatting scene.

    Returns
    -------
    images_out  : dict — normalized ColmapImage objects (deep-copied)
    points3D_out: dict — normalized, outlier-filtered ColmapPoint3D objects
    norm_info   : dict — {"centroid", "scale", "applied", "camera_radius",
                          "points_before", "points_after"}
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
        return images, points3D, {
            "centroid": np.zeros(3), "scale": 1.0,
            "applied": False, "camera_radius": 0.0,
        }

    centroid = centers.mean(axis=0)  # (3,)

    # ── FIX-1: compute scale from camera radius, not point-cloud max_abs ─────
    camera_offsets = centers - centroid                        # (N, 3)
    camera_radius  = float(np.max(np.linalg.norm(camera_offsets, axis=1)))

    if camera_radius < 1e-9:
        log.warning("[normalize_scene] Camera radius is near-zero. Skipping scale.")
        scale = 1.0
    else:
        scale = 1.0 / camera_radius

    log.info(
        "[normalize_scene] centroid=%s  camera_radius=%.4f  scale=%.6f",
        centroid.round(4).tolist(), camera_radius, scale,
    )

    # ── Step 2: apply to camera extrinsics ───────────────────────────────────
    # ColmapImage stores tvec (world→camera translation).
    # camera_center() = -R^T @ tvec.
    # new_center = (old_center - centroid) * scale
    # new_tvec   = -R @ new_center
    for img in images.values():
        R          = img.rotation_matrix()       # (3,3)
        old_center = img.camera_center()         # (3,)
        new_center = (old_center - centroid) * scale
        img.tvec   = (-R @ new_center).astype(np.float64)

    # ── FIX-2: filter COLMAP point-cloud outliers before applying scale ───────
    points_before = len(points3D)
    if points3D:
        xyz_all = np.array([pt.xyz for pt in points3D.values()], dtype=np.float64)
        dists   = np.linalg.norm(xyz_all - centroid, axis=1)
        p99     = np.percentile(dists, 99)

        outlier_ids = [
            pt_id for pt_id, pt in points3D.items()
            if np.linalg.norm(np.asarray(pt.xyz, dtype=np.float64) - centroid) >= p99
        ]
        for pt_id in outlier_ids:
            del points3D[pt_id]

        log.info(
            "[normalize_scene] Point cloud: %d → %d points after p99 outlier filter "
            "(removed %d; p99 dist=%.2f m)",
            points_before, len(points3D), len(outlier_ids), p99,
        )

    # ── Step 3: apply translation + scale to surviving 3-D points ────────────
    for pt in points3D.values():
        pt.xyz = ((np.asarray(pt.xyz, dtype=np.float64) - centroid) * scale)

    norm_info = {
        "centroid":      centroid,
        "scale":         scale,
        "applied":       True,
        "camera_radius": camera_radius,
        "points_before": points_before,
        "points_after":  len(points3D),
    }
    return images, points3D, norm_info


def scene_stats(images: Dict, points3D: Dict) -> Dict:
    """
    Return diagnostic statistics about the scene (before or after normalization).

    Keys returned
    -------------
    camera_extent : float  — median pairwise camera distance
    camera_radius : float  — max camera distance from centroid
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

    centroid = centers.mean(axis=0) if n_cam > 0 else np.zeros(3)
    camera_radius = float(np.max(np.linalg.norm(centers - centroid, axis=1))) \
        if n_cam > 0 else 0.0

    return {
        "camera_extent":  round(extent, 6),
        "camera_radius":  round(camera_radius, 6),
        "centroid":       centroid.round(6).tolist(),
        "max_abs_xyz":    float(np.max(np.abs(point_xyz))) if len(point_xyz) > 0 else 0.0,
        "n_cameras":      n_cam,
        "n_points":       len(point_xyz),
    }
