"""Optical-flow motion scoring reused by preprocessing and reports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .common import clamp01, image_files


def estimate_motion(image_dir: str | Path) -> float:
    """Return mean dense optical-flow magnitude across adjacent frames."""
    try:
        import cv2
    except ImportError:
        return 0.1

    frames = image_files(image_dir)
    if len(frames) < 2:
        return 0.1

    total_motion = 0.0
    count = 0
    for left, right in zip(frames, frames[1:]):
        img1 = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
        total_motion += float(magnitude.mean())
        count += 1

    motion = total_motion / max(count, 1)
    return 0.1 if motion < 0.01 else motion


def motion_score(raw_motion: float) -> float:
    """Score useful camera motion: too little hurts coverage; too much suggests blur/shake."""
    if raw_motion < 1.0:
        return clamp01(raw_motion / 1.0)
    if raw_motion <= 15.0:
        return 1.0
    return clamp01(1.0 - ((raw_motion - 15.0) / 35.0))


def analyze_motion(image_dir: str | Path) -> Dict:
    raw = estimate_motion(image_dir)
    return {
        "motion_score": motion_score(raw),
        "raw_motion": raw,
        "motion_quality": "low" if raw < 1.0 else ("high" if raw > 30.0 else "good"),
    }
