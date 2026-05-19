"""
dataset_quality.py
Pre-training dataset quality analysis.

Detects issues that will silently degrade training quality if not caught:
  - Motion blur (Laplacian variance too low)
  - Weak camera coverage (too few images)
  - Poor exposure consistency (high std-dev in brightness)
  - Low parallax (all cameras too similar — bad for 3DGS)
  - COLMAP registration quality

All checks return structured dicts so they can be serialised to metrics.json.
The validate() function returns (ok: bool, warnings: list[str], errors: list[str])
— callers can decide whether to hard-fail on errors or continue with warnings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_image_count(image_paths: List[str], min_images: int = 30) -> dict:
    n = len(image_paths)
    ok = n >= min_images
    return {
        "check":      "image_count",
        "ok":         ok,
        "count":      n,
        "min":        min_images,
        "message":    f"{n} images found" if ok else f"Only {n} images — recommend ≥{min_images} for stable COLMAP.",
    }


def check_blur(
    image_paths: List[str],
    threshold: float = 50.0,
    sample_n: int = 20,
) -> dict:
    """
    Estimate blur level via Laplacian variance on a sample of images.
    Returns the fraction of images estimated to be too blurry.
    """
    try:
        import cv2
    except ImportError:
        return {"check": "blur", "ok": True, "skipped": True, "message": "cv2 not available — blur check skipped."}

    sample = image_paths[::max(1, len(image_paths) // sample_n)][:sample_n]
    variances = []
    for p in sample:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                variances.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
        except Exception:
            pass

    if not variances:
        return {"check": "blur", "ok": True, "skipped": True, "message": "Could not read images for blur check."}

    mean_var   = float(np.mean(variances))
    n_blurry   = sum(1 for v in variances if v < threshold)
    blurry_pct = n_blurry / len(variances) * 100

    ok = blurry_pct < 30  # warn if >30% of sampled frames are blurry

    return {
        "check":       "blur",
        "ok":          ok,
        "mean_laplacian_var": round(mean_var, 1),
        "blurry_pct":  round(blurry_pct, 1),
        "threshold":   threshold,
        "n_sampled":   len(variances),
        "message": (
            f"Blur OK (mean Laplacian variance={mean_var:.0f})" if ok else
            f"{blurry_pct:.0f}% of sampled frames appear blurry (Laplacian<{threshold}). "
            "Blurry input degrades COLMAP matching and 3DGS quality."
        ),
    }


def check_exposure_consistency(
    image_paths: List[str],
    max_std_pct: float = 30.0,
    sample_n: int = 20,
) -> dict:
    """
    Check brightness consistency across a sample of images.
    High std-dev in mean brightness indicates mixed indoor/outdoor or
    poor auto-exposure, which causes COLMAP photometric failures.
    """
    try:
        import cv2
    except ImportError:
        return {"check": "exposure", "ok": True, "skipped": True, "message": "cv2 not available."}

    sample = image_paths[::max(1, len(image_paths) // sample_n)][:sample_n]
    means = []
    for p in sample:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                means.append(float(img.mean()))
        except Exception:
            pass

    if len(means) < 2:
        return {"check": "exposure", "ok": True, "skipped": True, "message": "Too few readable images for exposure check."}

    arr = np.array(means)
    std_pct = float(arr.std() / max(arr.mean(), 1e-6) * 100)
    ok = std_pct < max_std_pct

    return {
        "check":         "exposure",
        "ok":            ok,
        "brightness_std_pct": round(std_pct, 1),
        "max_std_pct":   max_std_pct,
        "n_sampled":     len(means),
        "message": (
            f"Exposure consistent (std={std_pct:.0f}%)" if ok else
            f"Exposure inconsistent: brightness std={std_pct:.0f}% > {max_std_pct}%. "
            "Mixed exposure causes colour banding in 3DGS renders."
        ),
    }


def check_colmap_registration(
    registered: int,
    total: int,
    min_ratio: float = 0.7,
    min_sparse_points: int = 1000,
    sparse_points: Optional[int] = None,
) -> dict:
    """Check COLMAP registration quality."""
    ratio = registered / max(total, 1)
    ok_ratio  = ratio >= min_ratio
    ok_points = (sparse_points is None) or (sparse_points >= min_sparse_points)
    ok = ok_ratio and ok_points

    msgs = []
    if not ok_ratio:
        msgs.append(f"Only {registered}/{total} images registered ({ratio*100:.0f}% < {min_ratio*100:.0f}% threshold).")
    if not ok_points and sparse_points is not None:
        msgs.append(f"Only {sparse_points:,} sparse points (need ≥{min_sparse_points:,}).")
    if ok:
        msgs.append(f"COLMAP OK: {registered}/{total} registered, {sparse_points or '?'} points.")

    return {
        "check":            "colmap_registration",
        "ok":               ok,
        "registered":       registered,
        "total":            total,
        "ratio":            round(ratio, 3),
        "sparse_points":    sparse_points,
        "message":          " ".join(msgs),
    }


def check_parallax(
    camera_positions: Optional[np.ndarray],
    min_spread_pct: float = 5.0,
) -> dict:
    """
    Estimate camera parallax from positions.

    Low parallax (all cameras in nearly the same location) leads to very
    poor depth estimation and flat 3DGS splats.

    camera_positions: (N, 3) float array of camera positions in world space.
    min_spread_pct:   minimum spread as % of scene diagonal to be "ok".
    """
    if camera_positions is None or len(camera_positions) < 2:
        return {"check": "parallax", "ok": True, "skipped": True, "message": "No camera positions available."}

    scene_min = camera_positions.min(axis=0)
    scene_max = camera_positions.max(axis=0)
    diagonal  = float(np.linalg.norm(scene_max - scene_min))
    spread    = diagonal

    # A single hand-held video sweep should have spread > 5% of scene extent
    # This is a heuristic — it catches the degenerate case of a static camera.
    scene_extent = float(np.linalg.norm(camera_positions - camera_positions.mean(axis=0), axis=1).max())
    spread_pct   = (diagonal / max(scene_extent, 1e-6)) * 100

    ok = spread_pct >= min_spread_pct

    return {
        "check":       "parallax",
        "ok":          ok,
        "spread_pct":  round(spread_pct, 1),
        "diagonal":    round(diagonal, 3),
        "message": (
            f"Parallax OK (camera spread={spread_pct:.0f}%)" if ok else
            f"Low parallax: camera spread={spread_pct:.0f}% (min {min_spread_pct}%). "
            "Static or near-static camera leads to flat, under-converged 3DGS."
        ),
    }


# ---------------------------------------------------------------------------
# Validate — aggregates all checks
# ---------------------------------------------------------------------------

def validate(
    image_paths: List[str],
    registered: Optional[int] = None,
    total_images: Optional[int] = None,
    sparse_points: Optional[int] = None,
    camera_positions: Optional[np.ndarray] = None,
    run_blur: bool = True,
    run_exposure: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """
    Run all dataset quality checks.

    Returns
    -------
    ok       : True if no hard errors
    warnings : list of warning strings (non-fatal)
    errors   : list of error strings (fatal — training will likely fail)
    """
    results = []

    results.append(check_image_count(image_paths))

    if run_blur and image_paths:
        results.append(check_blur(image_paths))

    if run_exposure and image_paths:
        results.append(check_exposure_consistency(image_paths))

    if registered is not None and total_images is not None:
        results.append(check_colmap_registration(registered, total_images, sparse_points=sparse_points))

    if camera_positions is not None:
        results.append(check_parallax(camera_positions))

    warnings_out: List[str] = []
    errors_out: List[str] = []

    for r in results:
        if r.get("skipped"):
            continue
        if not r["ok"]:
            msg = f"[{r['check']}] {r['message']}"
            # Image count and COLMAP failures are hard errors; others are warnings
            if r["check"] in ("image_count", "colmap_registration"):
                errors_out.append(msg)
            else:
                warnings_out.append(msg)

    all_ok = len(errors_out) == 0
    return all_ok, warnings_out, errors_out
