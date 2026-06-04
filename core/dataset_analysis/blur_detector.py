"""Blur scoring based on the existing MonoSplat Laplacian-variance heuristic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .common import clamp01, image_files, mean


def score_blur_value(laplacian_variance: float, threshold: float = 120.0) -> float:
    """Map Laplacian variance to a 0..1 sharpness score."""
    return clamp01(laplacian_variance / max(threshold, 1e-6))


def laplacian_variance(path: str | Path) -> float:
    """Compute the blur signal used by the legacy frame filter."""
    try:
        import cv2
    except ImportError:
        return 0.0

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def classify_blur(
    image_dir: str | Path,
    threshold: float = 120.0,
) -> List[Tuple[Path, float, bool]]:
    """Return `(path, laplacian_variance, is_blurry)` for each frame."""
    rows = []
    for path in image_files(image_dir):
        value = laplacian_variance(path)
        rows.append((path, value, value < threshold))
    return rows


def analyze_blur(image_dir: str | Path, threshold: float = 120.0) -> Dict:
    rows = classify_blur(image_dir, threshold=threshold)
    total = len(rows)
    blurry = sum(1 for _, _, is_blurry in rows if is_blurry)
    values = [value for _, value, _ in rows]
    sharp_ratio = 1.0 - (blurry / max(total, 1))
    variance_score = score_blur_value(mean(values), threshold=threshold)
    score = clamp01(0.7 * sharp_ratio + 0.3 * variance_score)
    return {
        "blur_score": score,
        "mean_laplacian_variance": mean(values),
        "threshold": threshold,
        "total_frames": total,
        "blurry_frames": blurry,
        "sharp_frames": total - blurry,
    }


def move_blurry_images(image_dir: str | Path, threshold: float = 120.0) -> int:
    """Legacy-compatible mutating filter: move blurry frames to `blurry/`."""
    import gc

    image_dir = Path(image_dir)
    rows = classify_blur(image_dir, threshold=threshold)
    if not rows:
        return 0

    bad_dir = image_dir / "blurry"
    bad_dir.mkdir(exist_ok=True)
    removed = kept = 0
    for path, _value, is_blurry in rows:
        gc.collect()
        if is_blurry:
            path.rename(bad_dir / path.name)
            removed += 1
        else:
            kept += 1
    print(f"[extract] Blur filter: {removed} blurry frames moved to '{bad_dir}', {kept} kept")
    return kept
