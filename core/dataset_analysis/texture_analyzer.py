"""Texture richness scoring based on the existing CLAHE + SIFT feature heuristic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .common import clamp01, image_files, mean


def feature_counts(image_dir: str | Path) -> List[Tuple[Path, int]]:
    try:
        import cv2
    except ImportError:
        return []

    try:
        detector = cv2.SIFT_create()
    except Exception:
        detector = cv2.ORB_create(nfeatures=2000)

    rows = []
    for path in image_files(image_dir):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append((path, 0))
            continue
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        rows.append((path, len(detector.detect(img, None))))
    return rows


def dynamic_feature_threshold(avg_features: float) -> int:
    """Existing dynamic threshold used by `filter_low_feature_frames`."""
    if avg_features < 300:
        return 10
    if avg_features < 1000:
        return 25
    return 50


def analyze_texture(image_dir: str | Path) -> Dict:
    rows = feature_counts(image_dir)
    counts = [count for _, count in rows]
    avg_features = mean(counts)
    threshold = dynamic_feature_threshold(avg_features)
    rich_frames = sum(1 for count in counts if count >= threshold)
    rich_ratio = rich_frames / max(len(counts), 1)
    density_score = clamp01(avg_features / 800.0)
    score = clamp01(0.65 * rich_ratio + 0.35 * density_score)
    return {
        "texture_score": score,
        "avg_features": avg_features,
        "feature_threshold": threshold,
        "total_frames": len(counts),
        "rich_texture_frames": rich_frames,
        "low_texture_frames": len(counts) - rich_frames,
    }


def remove_low_feature_frames(
    image_dir: str | Path,
    min_keep_ratio: float = 0.9,
) -> int:
    """Legacy-compatible mutating feature filter."""
    image_dir = Path(image_dir)
    counts = feature_counts(image_dir)
    if not counts:
        return 0

    feature_values = [n for _, n in counts]
    avg_features = mean(feature_values)
    threshold = dynamic_feature_threshold(avg_features)
    print(f"[extract] Feature filter: {len(counts)} frames | avg={avg_features:.0f} | threshold={threshold}")

    min_keep = max(1, int(len(counts) * min_keep_ratio))
    above_threshold = sum(1 for _, n in counts if n >= threshold)
    if above_threshold < min_keep:
        sorted_counts = sorted(counts, key=lambda x: x[1], reverse=True)
        threshold = sorted_counts[min_keep - 1][1]
        print(f"[extract] Relaxed threshold to {threshold} to keep {min_keep} frames")

    removed = 0
    for path, count in counts:
        if count < threshold:
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass

    kept = len(counts) - removed
    print(f"[extract] Feature filter: {removed} removed, {kept} kept")
    return kept
