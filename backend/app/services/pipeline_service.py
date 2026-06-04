"""
backend/app/services/pipeline_service.py
------------------------------------------
Thin service wrapper over scripts/pipeline.py.
Called by job_runner background tasks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger("monosplat.services.pipeline")


def process_video(
    video_path: str,
    output_root: str = "outputs",
    fps: float = 2.0,
    max_frames: int = 150,
    quality: str = "medium",
    use_gpu: bool = True,
    use_quality_gate: bool = False,
) -> dict:
    """
    Run the full local preprocessing pipeline.

    Returns dict: { frames, dataset, zip }
    """
    from scripts.pipeline import run_pipeline

    log.info("Starting pipeline for: %s", video_path)
    result = run_pipeline(
        video_path=video_path,
        output_root=output_root,
        fps=fps,
        max_frames=max_frames,
        quality=quality,
        use_gpu=use_gpu,
        use_quality_gate=use_quality_gate,
    )
    log.info("Pipeline complete: zip=%s", result.get("zip"))
    return result
