"""Pre-COLMAP reconstruction success prediction for MonoSplat."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


SCORE_KEYS = (
    "blur_score",
    "exposure_score",
    "motion_score",
    "coverage_score",
    "texture_score",
    "dynamic_object_score",
)


@dataclass
class ReconstructionSuccessPredictor:
    """Predict reconstruction success from dataset-analysis scores."""

    high_risk_threshold: float = 0.45
    medium_risk_threshold: float = 0.70

    def predict(
        self,
        analysis_report: Dict,
        output_path: Optional[str | Path] = None,
    ) -> Dict:
        scores = _extract_scores(analysis_report)
        probability = _weighted_probability(scores)
        risk_level = _risk_level(
            probability,
            self.high_risk_threshold,
            self.medium_risk_threshold,
        )
        issues = _risk_factors(scores)
        recommended_action = _recommended_action(risk_level, issues)

        report = {
            "generated_at": time.time(),
            "success_probability": probability,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "inputs": scores,
            "explanation": {
                "summary": _summary(risk_level, probability),
                "risk_factors": issues,
            },
        }

        if output_path is not None:
            save_prediction_report(report, output_path)
        return report


def _extract_scores(report: Dict) -> Dict[str, float]:
    return {key: _clamp01(float(report.get(key, 0.0))) for key in SCORE_KEYS}


def _weighted_probability(scores: Dict[str, float]) -> float:
    value = (
        0.20 * scores["blur_score"]
        + 0.15 * scores["exposure_score"]
        + 0.15 * scores["motion_score"]
        + 0.25 * scores["coverage_score"]
        + 0.20 * scores["texture_score"]
        + 0.05 * scores["dynamic_object_score"]
    )
    return round(_clamp01(value), 4)


def _risk_level(probability: float, high_risk_threshold: float, medium_risk_threshold: float) -> str:
    if probability < high_risk_threshold:
        return "high"
    if probability < medium_risk_threshold:
        return "medium"
    return "low"


def _risk_factors(scores: Dict[str, float]) -> list[str]:
    factors = []
    thresholds = {
        "blur_score": (0.55, "Excessive blur"),
        "exposure_score": (0.60, "Poor exposure balance"),
        "motion_score": (0.45, "Insufficient or unstable camera motion"),
        "coverage_score": (0.55, "Coverage score below threshold"),
        "texture_score": (0.55, "Low texture richness"),
        "dynamic_object_score": (0.55, "Dynamic-object risk detected"),
    }
    for key, (threshold, label) in thresholds.items():
        score = scores[key]
        if score < threshold:
            factors.append(f"{label}: {score:.2f} < {threshold:.2f}")
    return factors


def _recommended_action(risk_level: str, issues: Iterable[str]) -> str:
    issues = list(issues)
    if risk_level == "low":
        return "Proceed to COLMAP."
    if risk_level == "medium":
        if issues:
            return "Proceed with caution; improve capture if COLMAP registration is weak."
        return "Proceed with caution; monitor COLMAP registration and sparse point count."
    if issues:
        return "High risk before COLMAP. Improve the dataset first: " + "; ".join(issues[:3]) + "."
    return "High risk before COLMAP. Reshoot with more overlap, sharper frames, and richer texture."


def _summary(risk_level: str, probability: float) -> str:
    if risk_level == "low":
        return f"Dataset is likely reconstructable ({probability:.2f})."
    if risk_level == "medium":
        return f"Dataset may reconstruct but has quality risks ({probability:.2f})."
    return f"Dataset is high risk for reconstruction failure ({probability:.2f})."


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def save_prediction_report(report: Dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    tmp.replace(path)
    return path
