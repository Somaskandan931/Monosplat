"""
backend/app/services/prediction_service.py
-------------------------------------------
PredictionService — wraps core.quality_prediction.ReconstructionSuccessPredictor.

Converts a DatasetAnalysisPipeline report dict into a structured prediction
with risk level, recommended action, and per-factor explanations.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("monosplat.services.prediction")

_REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class PredictionService:
    """
    Predict reconstruction success probability from a quality report.

    Wraps:
        core.quality_prediction.ReconstructionSuccessPredictor
    """

    def __init__(
        self,
        high_risk_threshold: float = 0.45,
        medium_risk_threshold: float = 0.70,
    ) -> None:
        self.high_risk_threshold   = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold

    def predict(
        self,
        analysis_report: Dict[str, Any],
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction on an existing analysis report.

        Parameters
        ----------
        analysis_report : dict from DatasetAnalysisService.run()
        output_path     : optional path to persist the prediction JSON

        Returns
        -------
        dict with keys:
            success_probability, risk_level, recommended_action,
            inputs, explanation.{summary, risk_factors}
        """
        from core.quality_prediction import ReconstructionSuccessPredictor

        predictor = ReconstructionSuccessPredictor(
            high_risk_threshold=self.high_risk_threshold,
            medium_risk_threshold=self.medium_risk_threshold,
        )
        result = predictor.predict(analysis_report, output_path=output_path)

        log.info(
            "Prediction — probability=%.2f  risk=%s",
            result.get("success_probability", 0.0),
            result.get("risk_level", "unknown"),
        )
        return result
