"""Lightweight dynamic-object risk detector using optical-flow inconsistency."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .common import clamp01, image_files, mean, sample_files


def analyze_dynamic_objects(image_dir: str | Path) -> Dict:
    """Estimate static-scene quality from flow variance.

    This is intentionally lightweight: it flags unstable local motion patterns
    without introducing a detector dependency. A high score means the dataset
    appears mostly static.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"dynamic_object_score": 1.0, "flow_variance": 0.0, "dynamic_risk": "unknown"}

    frames = sample_files(image_files(image_dir), max_samples=30)
    if len(frames) < 2:
        return {"dynamic_object_score": 1.0, "flow_variance": 0.0, "dynamic_risk": "low"}

    variances = []
    for left, right in zip(frames, frames[1:]):
        img1 = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        variances.append(float(magnitude.var()))

    flow_variance = mean(variances)
    score = clamp01(1.0 - flow_variance / 120.0)
    risk = "low" if score >= 0.75 else ("medium" if score >= 0.45 else "high")
    return {
        "dynamic_object_score": score,
        "flow_variance": flow_variance,
        "dynamic_risk": risk,
    }
