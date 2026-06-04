"""Coverage estimation from frame count, motion, diversity, and texture signals."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .common import clamp01, image_files
from .motion_analyzer import motion_score


def estimate_coverage(
    image_dir: str | Path,
    raw_motion: float = 0.0,
    texture_score: float = 0.0,
) -> Dict:
    frames = image_files(image_dir)
    frame_count = len(frames)
    frame_score = clamp01(frame_count / 150.0)
    score = clamp01(
        0.45 * frame_score
        + 0.35 * motion_score(raw_motion)
        + 0.20 * texture_score
    )
    recommended_frames = 100
    if score >= 0.85:
        recommended_frames = min(300, max(150, frame_count))
    elif score >= 0.65:
        recommended_frames = 150
    elif score >= 0.45:
        recommended_frames = 200
    else:
        recommended_frames = 300
    return {
        "coverage_score": score,
        "frame_count": frame_count,
        "recommended_frames": recommended_frames,
    }
