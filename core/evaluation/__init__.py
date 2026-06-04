"""
core/evaluation/__init__.py
-----------------------------
MonoSplat Evaluation Framework — public API.

Auto-integration after training:
  from core.evaluation import EvaluationPipeline
  pipeline = EvaluationPipeline(run_dir, run_id)
  paths = pipeline.run_post_training(
      model=model,
      rendered_images=test_renders,
      gt_images=test_gts,
      training_result=training_result_dict,
  )

Standalone CLI:
  python -m core.evaluation --run-dir experiments/run_xyz --checkpoint ...

Comparison:
  from core.evaluation import compare_runs
  compare_runs(["experiments/run_a", "experiments/run_b"], output_dir="experiments")
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

log = logging.getLogger("monosplat.evaluation")

from .psnr              import PSNREvaluator
from .ssim              import SSIMEvaluator
from .lpips             import LPIPSEvaluator
from .fps_benchmark     import FPSBenchmark
from .evaluation_report import EvaluationReport

__all__ = [
    "PSNREvaluator",
    "SSIMEvaluator",
    "LPIPSEvaluator",
    "FPSBenchmark",
    "EvaluationReport",
    "EvaluationPipeline",
    "compare_runs",
]


class EvaluationPipeline:
    """
    Orchestrates all MonoSplat evaluators and writes the three report formats
    to the run's experiment directory.

    Parameters
    ----------
    run_dir   : experiment directory (e.g. experiments/run_20260603_161200)
    run_id    : string ID for the run
    device    : 'cuda' / 'cpu' / None (auto)
    lpips_net : LPIPS backbone ('vgg', 'alex', 'squeeze')
    """

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        device: Optional[str] = None,
        lpips_net: str = "vgg",
    ) -> None:
        self.run_dir   = Path(run_dir)
        self.run_id    = run_id
        self.device    = device
        self.lpips_net = lpips_net

        self._psnr  = PSNREvaluator()
        self._ssim  = SSIMEvaluator()
        self._lpips = LPIPSEvaluator(net=lpips_net, device=device)
        self._fps   = FPSBenchmark()

    # ── Main post-training integration point ───────────────────────────────

    def run_post_training(
        self,
        *,
        model=None,
        rendered_images: Optional[Sequence] = None,
        gt_images: Optional[Sequence] = None,
        image_names: Optional[Sequence[str]] = None,
        training_result: Optional[Dict[str, Any]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[str] = None,
        skip_fps: bool = False,
    ) -> Dict[str, str]:
        """
        Run the full evaluation suite and save reports.

        Called automatically by TrainingService.run() immediately after
        training completes. Can also be called manually.

        Parameters
        ----------
        model            : loaded GaussianModel (for FPS benchmark)
        rendered_images  : test-set renders (numpy arrays or paths)
        gt_images        : test-set ground truths (numpy arrays or paths)
        image_names      : names for per-image rows in the report
        training_result  : dict from TrainingService.run() with keys:
                           run_id, model_path, checkpoint_path, final_metrics
        config_snapshot  : config dict logged at training start
        dataset_path     : path to frame directory
        skip_fps         : set True to skip FPS benchmark (e.g. no GPU)

        Returns
        -------
        dict with "json", "md", "html" — paths to the three report files.
        """
        t_result = training_result or {}
        report   = EvaluationReport(
            run_id=self.run_id,
            run_dir=self.run_dir,
            dataset_path=dataset_path,
            config_snapshot=config_snapshot,
        )

        # ── Image quality metrics ──────────────────────────────────────────
        if rendered_images and gt_images:
            n = min(len(rendered_images), len(gt_images))
            rend_arr = list(rendered_images[:n])
            gt_arr   = list(gt_images[:n])
            names    = list(image_names[:n]) if image_names else None

            log.info("Evaluating PSNR over %d frames …", n)
            report.set_psnr(self._psnr.evaluate_set(rend_arr, gt_arr, names))

            log.info("Evaluating SSIM …")
            report.set_ssim(self._ssim.evaluate_set(rend_arr, gt_arr, names))

            log.info("Evaluating LPIPS …")
            report.set_lpips(self._lpips.evaluate_set(rend_arr, gt_arr, names))
        else:
            log.warning("No rendered/GT image pairs supplied — image quality metrics skipped.")
            # Write placeholder so the report is still valid JSON
            for setter in (report.set_psnr, report.set_ssim, report.set_lpips):
                setter({"mean": None, "min": None, "max": None, "std": None, "count": 0})

        # ── FPS benchmark ──────────────────────────────────────────────────
        if model is not None and not skip_fps:
            log.info("Running FPS benchmark …")
            try:
                fps_result = self._fps.run(model, device=self.device)
                report.set_fps(fps_result)
            except Exception as exc:
                log.warning("FPS benchmark failed: %s", exc)
                report.set_fps({"primary_fps": None, "error": str(exc)})
        else:
            report.set_fps({"primary_fps": None, "note": "benchmark skipped"})

        # ── Training stats ─────────────────────────────────────────────────
        final_metrics = t_result.get("final_metrics") or {}
        report.set_training(
            duration_seconds=t_result.get("duration_seconds"),
            iterations=t_result.get("iterations") or final_metrics.get("iteration"),
            final_loss=t_result.get("final_loss") or final_metrics.get("loss"),
            n_gaussians=t_result.get("n_gaussians") or final_metrics.get("n_gaussians"),
            model_path=t_result.get("model_path"),
            checkpoint_path=t_result.get("checkpoint_path"),
        )

        paths = report.save()
        log.info(
            "Evaluation complete — PSNR=%.2f SSIM=%.4f LPIPS=%.4f",
            report._data.get("psnr", {}).get("mean") or 0.0,
            report._data.get("ssim", {}).get("mean") or 0.0,
            report._data.get("lpips", {}).get("mean") or 0.0,
        )
        return paths

    # ── Quick single-metric helpers ────────────────────────────────────────

    def psnr(self, rendered: np.ndarray, gt: np.ndarray) -> float:
        return self._psnr.compute(rendered, gt)

    def ssim(self, rendered: np.ndarray, gt: np.ndarray) -> float:
        return self._ssim.compute(rendered, gt)

    def lpips(self, rendered: np.ndarray, gt: np.ndarray) -> float:
        return self._lpips.compute(rendered, gt)


# ── Standalone comparison helper ───────────────────────────────────────────────

def compare_runs(
    run_dirs: Sequence[str | Path],
    output_dir: Optional[str | Path] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare evaluation reports from multiple experiment directories.

    Parameters
    ----------
    run_dirs   : list of experiment directories, each containing evaluation_report.json
    output_dir : where to write comparison_{ts}.html (defaults to first run_dir)
    labels     : human-readable names (defaults to directory names)

    Returns
    -------
    dict from EvaluationReport.compare()
    """
    reports: List[Dict] = []
    resolved_labels: List[str] = []

    for i, rd in enumerate(run_dirs):
        rd   = Path(rd)
        rpt  = rd / "evaluation_report.json"
        label = (labels[i] if labels and i < len(labels) else rd.name)
        if not rpt.exists():
            log.warning("No evaluation_report.json in %s — skipping", rd)
            continue
        with open(rpt) as fh:
            reports.append(json.load(fh))
        resolved_labels.append(label)

    if not reports:
        raise FileNotFoundError("No evaluation_report.json files found in any of the run directories.")

    out_dir = output_dir or run_dirs[0]
    return EvaluationReport.compare(reports, output_dir=out_dir, labels=resolved_labels)
