"""
metrics.py
Structured pipeline metrics — collected at every stage and persisted to
work/<job_id>/metrics.json for deterministic debugging.

Usage
-----
    from src.utils.metrics import PipelineMetrics

    m = PipelineMetrics(job_id="abc123", work_dir="work")
    m.set_frame_metrics(frame_count=200, filtered_frames=120, blur_mean=95.3)
    m.set_reconstruction_metrics(registered_images=110, total_images=120,
                                  sparse_points=8400, reprojection_error=0.45)
    m.set_training_metrics(initial_gaussians=8400, final_gaussians=82000,
                            nan_iterations=2, loss_curve=[0.9, 0.4, 0.18])
    m.save()  # writes work/abc123/metrics.json

Design rules
------------
- Every field has a default of None (not yet populated).
- save() writes atomically (tmp → rename) to avoid partial-write corruption.
- All numeric types are plain Python (int / float) so json.dumps works without
  a custom encoder.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineMetrics:
    # ---- Identity --------------------------------------------------------
    job_id: str = ""
    run_timestamp: float = field(default_factory=time.time)
    pipeline_version: str = "3.1.0"

    # ---- Frame extraction ------------------------------------------------
    frame_count: Optional[int] = None          # raw frames from video
    filtered_frames: Optional[int] = None      # after blur / duplicate rejection
    blur_mean: Optional[float] = None          # mean Laplacian variance across frames
    blur_rejected: Optional[int] = None        # frames dropped for blur
    duplicate_rejected: Optional[int] = None   # frames dropped as near-duplicates
    exposure_ok: Optional[bool] = None         # True if exposure stats pass

    # ---- COLMAP reconstruction -------------------------------------------
    total_images: Optional[int] = None
    registered_images: Optional[int] = None
    registration_ratio: Optional[float] = None
    sparse_points: Optional[int] = None
    mean_reprojection_error: Optional[float] = None
    colmap_retried: bool = False               # True if low-quality retry was triggered

    # ---- Gaussian initialisation -----------------------------------------
    initial_gaussians: Optional[int] = None

    # ---- Training --------------------------------------------------------
    final_gaussians: Optional[int] = None
    nan_iterations: Optional[int] = None        # iterations skipped due to NaN loss
    loss_curve: List[float] = field(default_factory=list)
    final_loss: Optional[float] = None
    psnr_final: Optional[float] = None
    ssim_final: Optional[float] = None

    # ---- Outcome ---------------------------------------------------------
    status: str = "pending"                     # pending | success | failed
    failure_reason: Optional[str] = None
    wall_seconds: Optional[float] = None

    def set_frame_metrics(
        self,
        frame_count: int,
        filtered_frames: int,
        blur_mean: Optional[float] = None,
        blur_rejected: Optional[int] = None,
        duplicate_rejected: Optional[int] = None,
        exposure_ok: Optional[bool] = None,
    ) -> None:
        """Populate frame-extraction stage metrics."""
        self.frame_count = frame_count
        self.filtered_frames = filtered_frames
        self.blur_mean = blur_mean
        self.blur_rejected = blur_rejected
        self.duplicate_rejected = duplicate_rejected
        self.exposure_ok = exposure_ok

    def set_reconstruction_metrics(
        self,
        registered_images: int,
        total_images: int,
        sparse_points: int,
        reprojection_error: Optional[float] = None,
        retried: bool = False,
    ) -> None:
        """Populate COLMAP reconstruction stage metrics."""
        self.registered_images = registered_images
        self.total_images = total_images
        self.registration_ratio = registered_images / max(total_images, 1)
        self.sparse_points = sparse_points
        self.mean_reprojection_error = reprojection_error
        self.colmap_retried = retried

    def set_training_metrics(
        self,
        initial_gaussians: int,
        final_gaussians: int,
        nan_iterations: int,
        loss_curve: List[float],
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
    ) -> None:
        """Populate training stage metrics."""
        self.initial_gaussians = initial_gaussians
        self.final_gaussians = final_gaussians
        self.nan_iterations = nan_iterations
        self.loss_curve = loss_curve
        self.final_loss = loss_curve[-1] if loss_curve else None
        self.psnr_final = psnr
        self.ssim_final = ssim

    def mark_success(self, wall_seconds: Optional[float] = None) -> None:
        self.status = "success"
        self.wall_seconds = wall_seconds

    def mark_failed(self, reason: str, wall_seconds: Optional[float] = None) -> None:
        self.status = "failed"
        self.failure_reason = reason
        self.wall_seconds = wall_seconds

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, work_dir: Optional[str] = None) -> Path:
        """
        Write metrics.json atomically to work/<job_id>/metrics.json.

        If work_dir is provided, uses that as the root.
        Otherwise writes to the current working directory under <job_id>/metrics.json.
        """
        if work_dir:
            out_dir = Path(work_dir) / self.job_id
        else:
            out_dir = Path("work") / self.job_id

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
        tmp_path = out_dir / "metrics.json.tmp"

        payload = self.to_dict()
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        tmp_path.rename(out_path)  # atomic on POSIX

        print(f"[metrics] Saved → {out_path}")
        return out_path

    @classmethod
    def load(cls, path: str) -> "PipelineMetrics":
        """Load a metrics.json file back into a PipelineMetrics instance."""
        with open(path) as f:
            data = json.load(f)
        m = cls()
        for k, v in data.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m

    def summary(self) -> str:
        """One-line human-readable summary for log files."""
        return (
            f"[metrics] job={self.job_id} status={self.status} "
            f"frames={self.filtered_frames}/{self.frame_count} "
            f"registered={self.registered_images}/{self.total_images} "
            f"points={self.sparse_points} "
            f"gaussians={self.initial_gaussians}->{self.final_gaussians} "
            f"nan_iters={self.nan_iterations} "
            f"loss={self.final_loss:.4f if self.final_loss else 'N/A'}"
        )
