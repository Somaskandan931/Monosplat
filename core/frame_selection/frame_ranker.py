"""Frame ranking from feature density, blur, motion, and diversity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.dataset_analysis.blur_detector import laplacian_variance, score_blur_value
from core.dataset_analysis.common import clamp01, image_files
from core.frame_selection.duplicate_remover import histogram_similarity


@dataclass
class FrameScore:
    path: Path
    index: int
    score: float
    orb_features: int
    feature_score: float
    blur_score: float
    motion_score: float
    diversity_score: float


def orb_feature_count(path: str | Path, nfeatures: int = 2000) -> int:
    try:
        import cv2
    except ImportError:
        return 0

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return len(orb.detect(img, None))


def optical_flow_between(left: Optional[Path], right: Optional[Path]) -> float:
    if left is None or right is None:
        return 0.0
    try:
        import cv2
    except ImportError:
        return 0.0

    img1 = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 0.0
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
    return float(magnitude.mean())


def motion_quality_score(raw_motion: float) -> float:
    if raw_motion <= 0:
        return 0.3
    if raw_motion < 1.0:
        return clamp01(raw_motion)
    if raw_motion <= 15.0:
        return 1.0
    return clamp01(1.0 - ((raw_motion - 15.0) / 35.0))


def rank_frames(image_dir: str | Path) -> List[FrameScore]:
    frames = image_files(image_dir)
    ranked: List[FrameScore] = []
    for idx, frame in enumerate(frames):
        features = orb_feature_count(frame)
        feature_score = clamp01(features / 1000.0)
        blur_score = score_blur_value(laplacian_variance(frame), threshold=120.0)
        prev_frame = frames[idx - 1] if idx > 0 else None
        next_frame = frames[idx + 1] if idx + 1 < len(frames) else None
        raw_motion = max(
            optical_flow_between(prev_frame, frame),
            optical_flow_between(frame, next_frame),
        )
        motion_score = motion_quality_score(raw_motion)
        if prev_frame is None:
            diversity_score = 1.0
        else:
            diversity_score = clamp01(1.0 - histogram_similarity(prev_frame, frame))
        score = (
            0.35 * feature_score
            + 0.25 * blur_score
            + 0.20 * motion_score
            + 0.20 * diversity_score
        )
        ranked.append(FrameScore(
            path=frame,
            index=idx,
            score=round(score, 6),
            orb_features=features,
            feature_score=round(feature_score, 6),
            blur_score=round(blur_score, 6),
            motion_score=round(motion_score, 6),
            diversity_score=round(diversity_score, 6),
        ))
    return ranked
