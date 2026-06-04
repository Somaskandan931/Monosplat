"""Exposure scoring extracted from the existing preprocessing diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .common import clamp01, image_files, mean


def analyze_exposure(
    image_dir: str | Path,
    overexposed_thresh: float = 245.0,
    underexposed_thresh: float = 10.0,
) -> Dict:
    try:
        import cv2
    except ImportError:
        return {
            "exposure_score": 0.0,
            "total_frames": 0,
            "overexposed": 0,
            "underexposed": 0,
            "ok": 0,
            "mean_intensity": 0.0,
        }

    total = overexposed = underexposed = 0
    intensities = []
    for path in image_files(image_dir):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        total += 1
        value = float(img.mean())
        intensities.append(value)
        if value >= overexposed_thresh:
            overexposed += 1
        elif value <= underexposed_thresh:
            underexposed += 1

    ok = total - overexposed - underexposed
    ok_ratio = ok / max(total, 1)
    center_score = 1.0 - abs(mean(intensities, 127.5) - 127.5) / 127.5
    score = clamp01(0.75 * ok_ratio + 0.25 * center_score)
    return {
        "exposure_score": score,
        "total_frames": total,
        "overexposed": overexposed,
        "underexposed": underexposed,
        "ok": ok,
        "mean_intensity": mean(intensities),
    }
