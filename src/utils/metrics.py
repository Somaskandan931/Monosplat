"""
metrics.py
Structured pipeline metrics — collected at every stage and persisted to
work/<job_id>/metrics.json for deterministic debugging.

Usage
-----
    from src.utils.metrics import PipelineMetrics, TrainingMetricsLog

    # Pipeline-level summary (one per job):
    m = PipelineMetrics(job_id="abc123", work_dir="work")
    m.set_training_metrics(initial_gaussians=50000, final_gaussians=51230,
                            nan_iterations=0, loss_curve=[0.9, 0.4, 0.18])
    m.save()

    # Per-iteration log (appended during training):
    log = TrainingMetricsLog(path="logs/metrics.json")
    log.record(iteration=500, loss=0.24, n_gaussians=52000)
    log.flush()  # atomic write

    # Cross-validate after training:
    log.validate_final_loss(trainer.last_metrics["final_loss"])

Design rules
------------
- Every field defaults to None (not yet populated).
- save() writes atomically (tmp → rename) to avoid partial-write corruption.
- All numeric types are plain Python (int / float) — json.dumps works without
  a custom encoder.
- TrainingMetricsLog is the single authoritative per-iteration series.
  validate_final_loss() catches the stale-metric bug where the summary
  reads an earlier entry while the trainer reports a different last value.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# PipelineMetrics — one record per job
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    # ---- Identity --------------------------------------------------------
    job_id: str = ""
    run_timestamp: float = field(default_factory=time.time)
    pipeline_version: str = "3.2.0"

    # ---- Frame extraction ------------------------------------------------
    frame_count: Optional[int] = None
    filtered_frames: Optional[int] = None
    blur_mean: Optional[float] = None
    blur_rejected: Optional[int] = None
    duplicate_rejected: Optional[int] = None
    exposure_ok: Optional[bool] = None

    # ---- COLMAP reconstruction -------------------------------------------
    total_images: Optional[int] = None
    registered_images: Optional[int] = None
    registration_ratio: Optional[float] = None
    sparse_points: Optional[int] = None
    mean_reprojection_error: Optional[float] = None
    colmap_retried: bool = False

    # ---- Gaussian initialisation -----------------------------------------
    initial_gaussians: Optional[int] = None

    # ---- Training --------------------------------------------------------
    final_gaussians: Optional[int] = None
    nan_iterations: Optional[int] = None
    loss_curve: List[float] = field(default_factory=list)
    # final_loss is ALWAYS loss_curve[-1] — never a stale smoothed value
    final_loss: Optional[float] = None
    psnr_final: Optional[float] = None
    ssim_final: Optional[float] = None

    # ---- VRAM ------------------------------------------------------------
    peak_vram_gb: Optional[float] = None
    final_vram_alloc_gb: Optional[float] = None

    # ---- Outcome ---------------------------------------------------------
    status: str = "pending"
    failure_reason: Optional[str] = None
    wall_seconds: Optional[float] = None

    # ------------------------------------------------------------------

    def set_frame_metrics(
        self,
        frame_count: int,
        filtered_frames: int,
        blur_mean: Optional[float] = None,
        blur_rejected: Optional[int] = None,
        duplicate_rejected: Optional[int] = None,
        exposure_ok: Optional[bool] = None,
    ) -> None:
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
        peak_vram_gb: Optional[float] = None,
        final_vram_alloc_gb: Optional[float] = None,
    ) -> None:
        self.initial_gaussians = initial_gaussians
        self.final_gaussians = final_gaussians
        self.nan_iterations = nan_iterations
        self.loss_curve = [float(v) for v in loss_curve]
        # final_loss is ALWAYS the last real entry — never stale
        self.final_loss = self.loss_curve[-1] if self.loss_curve else None
        self.psnr_final = psnr
        self.ssim_final = ssim
        self.peak_vram_gb = peak_vram_gb
        self.final_vram_alloc_gb = final_vram_alloc_gb

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
        """Write metrics.json atomically."""
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
        tmp_path.rename(out_path)

        print(f"[metrics] Saved → {out_path}")
        return out_path

    @classmethod
    def load(cls, path: str) -> "PipelineMetrics":
        with open(path) as f:
            data = json.load(f)
        m = cls()
        for k, v in data.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m

    def summary(self) -> str:
        loss_str = f"{self.final_loss:.4f}" if self.final_loss is not None else "N/A"
        return (
            f"[metrics] job={self.job_id} status={self.status} "
            f"frames={self.filtered_frames}/{self.frame_count} "
            f"registered={self.registered_images}/{self.total_images} "
            f"points={self.sparse_points} "
            f"gaussians={self.initial_gaussians}->{self.final_gaussians} "
            f"nan_iters={self.nan_iterations} "
            f"loss={loss_str}"
        )


# ---------------------------------------------------------------------------
# TrainingMetricsLog — authoritative per-iteration record
# ---------------------------------------------------------------------------

class TrainingMetricsLog:
    """
    Append-only per-iteration metrics log.

    This is the single source of truth for training metrics.
    It writes atomically to disk and exposes validate_final_loss()
    to catch the stale-metric inconsistency described in the hardening spec.

    The root cause of the "Final loss: 0.2218 vs 0.3142" divergence was that
    the in-memory metrics_log list was flushed mid-training (capturing an
    earlier entry) but the summary cell read from the file rather than the
    live trainer state.  This class eliminates that ambiguity by making the
    log the canonical source and validating it against the trainer at the end.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        self._dirty: bool = False

        # Load existing entries if resuming
        if self._path.exists():
            try:
                with open(self._path) as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self._entries = loaded
            except (json.JSONDecodeError, OSError):
                self._entries = []

    # ------------------------------------------------------------------

    def record(
        self,
        iteration: int,
        loss: float,
        n_gaussians: int,
        elapsed_s: float = 0.0,
        nan_count: int = 0,
        oom_count: int = 0,
        vram_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Append one iteration record. Does NOT flush to disk."""
        entry: Dict[str, Any] = {
            "iteration":   iteration,
            "loss":        round(float(loss), 6),
            "n_gaussians": int(n_gaussians),
            "elapsed_s":   round(float(elapsed_s), 1),
            "nan_count":   int(nan_count),
            "oom_count":   int(oom_count),
        }
        if vram_stats:
            entry.update(vram_stats)
        self._entries.append(entry)
        self._dirty = True

    def flush(self) -> None:
        """Atomically write all entries to disk."""
        if not self._dirty:
            return
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._entries, f, indent=2)
        tmp.rename(self._path)
        self._dirty = False

    def final_loss(self) -> Optional[float]:
        """Return the loss from the last recorded entry, or None."""
        if not self._entries:
            return None
        return self._entries[-1]["loss"]

    def validate_final_loss(self, trainer_final_loss: float, tol: float = 1e-4) -> None:
        """
        Assert that the log's final_loss matches the trainer's reported final_loss.

        Raises AssertionError with a diagnostic message if they disagree beyond tol.
        This catches the stale-metric bug where the metrics summary reads an
        earlier entry while the trainer reports a different final value.
        """
        log_last = self.final_loss()
        if log_last is None:
            return  # Nothing logged — nothing to validate

        diff = abs(log_last - trainer_final_loss)
        if diff > tol:
            raise AssertionError(
                f"[metrics] Final loss inconsistency detected!\n"
                f"  Trainer reports  : {trainer_final_loss:.6f}\n"
                f"  Log last entry   : {log_last:.6f}\n"
                f"  Difference       : {diff:.6f}  (tolerance {tol})\n"
                "This usually means the log was flushed before the final step\n"
                "completed, or loss_val was captured from a NaN-skipped iter.\n"
                "Fix: call log.record(final_iter, trainer.last_loss) then log.flush()\n"
                "immediately after the training loop exits."
            )

    def loss_curve(self) -> List[float]:
        return [e["loss"] for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)
