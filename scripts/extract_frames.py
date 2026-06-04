"""
scripts/extract_frames.py
-------------------------
Re-export shim — implementation lives in src/preprocessing/extract_frames.py.

Import from here for convenience; the real logic is in src/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessing.extract_frames import (  # noqa: E402, F401
    copy_images,
    estimate_motion,
    extract_from_video,
    filter_blurry_images,
    filter_low_feature_frames,
    get_video_info,
    run_smart_frame_selection,
    validate_exposure,
    validate_image_resolution,
    validate_images,
)

__all__ = [
    "extract_from_video",
    "copy_images",
    "validate_images",
    "validate_image_resolution",
    "get_video_info",
    "filter_low_feature_frames",
    "filter_blurry_images",
    "estimate_motion",
    "validate_exposure",
    "run_smart_frame_selection",
]
