"""
backend/app/services/dataset_analysis_service.py
--------------------------------------------------
DatasetAnalysisService — wraps core.dataset_analysis.DatasetAnalysisPipeline
for use from FastAPI routes and background workers.

This service is the API boundary: routes pass validated Pydantic data in,
the service calls the existing MonoSplat core, persists results, and returns
plain dicts suitable for JSON serialisation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("monosplat.services.dataset_analysis")

# ── Ensure src/ is importable from the service layer ─────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC       = _REPO_ROOT / "src"
for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class DatasetAnalysisService:
    """
    Run pre-COLMAP dataset quality analysis on a directory of frames.

    Wraps:
        core.dataset_analysis.DatasetAnalysisPipeline
    """

    def __init__(self, blur_threshold: float = 120.0) -> None:
        self.blur_threshold = blur_threshold

    # ── Main entry point (subprocess-safe, no SQLAlchemy) ─────────────────────

    def run(
        self,
        image_dir: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Analyse *image_dir* and return the quality report dict.

        Parameters
        ----------
        image_dir   : directory of JPEG/PNG frames
        output_path : if set, persist report to this JSON file

        Returns
        -------
        dict matching DatasetAnalysisPipeline.analyze() schema:
            blur_score, exposure_score, motion_score, coverage_score,
            texture_score, dynamic_object_score, success_probability, …
        """
        from core.dataset_analysis import DatasetAnalysisPipeline

        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        n_images = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
        if n_images == 0:
            raise ValueError(f"No JPEG/PNG frames found in {image_dir}")

        log.info("Analysing %d frames in %s", n_images, image_dir)

        pipeline = DatasetAnalysisPipeline(blur_threshold=self.blur_threshold)
        report   = pipeline.analyze(image_dir, output_path=output_path)

        log.info(
            "Analysis complete — success_probability=%.2f, blur=%.2f",
            report.get("success_probability", 0.0),
            report.get("blur_score", 0.0),
        )
        return report

    # ── Convenience: run and persist automatically ────────────────────────────

    @staticmethod
    def default_output_path(image_dir: str | Path) -> Path:
        """Return a sensible default path for the quality report JSON."""
        return Path(image_dir).parent / "quality_report.json"
