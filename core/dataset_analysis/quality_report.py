"""Unified pre-COLMAP dataset analysis pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core.quality_prediction import ReconstructionSuccessPredictor

from .blur_detector import analyze_blur
from .coverage_estimator import estimate_coverage
from .dynamic_object_detector import analyze_dynamic_objects
from .exposure_analyzer import analyze_exposure
from .motion_analyzer import analyze_motion
from .texture_analyzer import analyze_texture


@dataclass
class DatasetAnalysisPipeline:
    """Run all dataset diagnostics and persist a quality report."""

    blur_threshold: float = 120.0

    def analyze(
        self,
        image_dir: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict:
        image_dir = Path(image_dir)
        blur = analyze_blur(image_dir, threshold=self.blur_threshold)
        exposure = analyze_exposure(image_dir)
        motion = analyze_motion(image_dir)
        texture = analyze_texture(image_dir)
        coverage = estimate_coverage(
            image_dir,
            raw_motion=motion["raw_motion"],
            texture_score=texture["texture_score"],
        )
        dynamic = analyze_dynamic_objects(image_dir)

        report = {
            "dataset_path": str(image_dir),
            "generated_at": time.time(),
            "blur_score": float(blur["blur_score"]),
            "exposure_score": float(exposure["exposure_score"]),
            "motion_score": float(motion["motion_score"]),
            "coverage_score": float(coverage["coverage_score"]),
            "texture_score": float(texture["texture_score"]),
            "dynamic_object_score": float(dynamic["dynamic_object_score"]),
            "recommended_frames": int(coverage["recommended_frames"]),
            "details": {
                "blur": blur,
                "exposure": exposure,
                "motion": motion,
                "coverage": coverage,
                "texture": texture,
                "dynamic_objects": dynamic,
            },
        }
        prediction = ReconstructionSuccessPredictor().predict(report)
        report["success_probability"] = float(prediction["success_probability"])

        if output_path is not None:
            save_quality_report(report, output_path)
        return report


def save_quality_report(report: Dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    tmp.replace(path)
    return path
