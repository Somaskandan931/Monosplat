"""Near-duplicate detection reused by smart frame selection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

from core.dataset_analysis.common import image_files


def compute_histogram(path: str | Path):
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((8 * 8 * 8,), dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def histogram_similarity(left: str | Path, right: str | Path) -> float:
    import cv2

    return float(cv2.compareHist(
        compute_histogram(left),
        compute_histogram(right),
        cv2.HISTCMP_CORREL,
    ))


def find_duplicate_frames(
    image_dir: str | Path,
    hist_threshold: float = 0.96,
) -> Dict:
    """Find adjacent near-duplicate frames without mutating the dataset."""
    frames = image_files(image_dir)
    if len(frames) < 2:
        return {"duplicates": [], "kept": [str(p) for p in frames], "duplicate_ratio": 0.0}

    kept: List[Path] = [frames[0]]
    duplicates: List[Path] = []
    ref = frames[0]
    for frame in frames[1:]:
        similarity = histogram_similarity(ref, frame)
        if similarity >= hist_threshold:
            duplicates.append(frame)
        else:
            kept.append(frame)
            ref = frame

    return {
        "duplicates": [str(p) for p in duplicates],
        "kept": [str(p) for p in kept],
        "duplicate_ratio": len(duplicates) / max(len(frames), 1),
    }


def move_unselected_frames(image_dir: str | Path, selected: Set[Path]) -> int:
    """Move frames not in `selected` out of the COLMAP-visible top-level folder."""
    image_dir = Path(image_dir)
    reject_dir = image_dir / "frame_selection_rejected"
    reject_dir.mkdir(exist_ok=True)
    selected_resolved = {p.resolve() for p in selected}
    moved = 0
    for frame in image_files(image_dir):
        if frame.resolve() not in selected_resolved:
            frame.rename(reject_dir / frame.name)
            moved += 1
    return moved
