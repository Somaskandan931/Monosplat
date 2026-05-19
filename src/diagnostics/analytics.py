"""
analytics.py
Training analytics — EMA loss, convergence diagnostics, Gaussian growth rate,
VRAM trends, and text/JSON summary generation.

These are non-training-path utilities: they read the metrics log and produce
reports without touching model state or CUDA memory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------

def ema(values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Exponential moving average.

    alpha=0.05 gives a ~20-step smoothing window — good for loss curves.
    Returns a list of the same length as values.
    """
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def convergence_rate(losses: List[float], window: int = 100) -> float:
    """
    Estimate convergence rate as the relative loss reduction per 100 iterations
    over the last `window` entries.

    Returns a value in (-inf, 0]: negative means improving.
    Returns 0.0 if there's insufficient data.
    """
    if len(losses) < window:
        return 0.0
    tail = losses[-window:]
    if tail[0] <= 0:
        return 0.0
    return (tail[-1] - tail[0]) / tail[0]


# ---------------------------------------------------------------------------
# MetricsAnalyzer
# ---------------------------------------------------------------------------

class MetricsAnalyzer:
    """
    Reads a training metrics.json and produces analytics.

    Usage
    -----
        analyzer = MetricsAnalyzer.from_file("logs/metrics.json")
        report   = analyzer.summary_report()
        analyzer.save_analytics("logs/analytics.json")
    """

    def __init__(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = entries

    @classmethod
    def from_file(cls, path: str) -> "MetricsAnalyzer":
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}, got {type(data)}")
        return cls(data)

    # ------------------------------------------------------------------

    def losses(self) -> List[float]:
        return [e["loss"] for e in self._entries]

    def iterations(self) -> List[int]:
        return [e["iteration"] for e in self._entries]

    def gaussian_counts(self) -> List[int]:
        return [e.get("n_gaussians", 0) for e in self._entries]

    def vram_alloc(self) -> List[float]:
        return [e.get("vram_alloc_gb", 0.0) for e in self._entries]

    def vram_peak(self) -> List[float]:
        # Use vram_peak_gb if present (hardened path), else fall back to alloc
        return [e.get("vram_peak_gb", e.get("vram_alloc_gb", 0.0)) for e in self._entries]

    # ------------------------------------------------------------------

    def summary_report(self) -> Dict[str, Any]:
        """Return a structured dict summarising the training run."""
        if not self._entries:
            return {"error": "no entries"}

        losses  = self.losses()
        iters   = self.iterations()
        n_gauss = self.gaussian_counts()
        vrams   = self.vram_alloc()
        v_peaks = self.vram_peak()

        smoothed  = ema(losses)
        conv_rate = convergence_rate(smoothed, window=min(100, len(smoothed)))

        first, last = self._entries[0], self._entries[-1]

        # Gaussian growth diagnostics
        gauss_start   = n_gauss[0] if n_gauss else 0
        gauss_end     = n_gauss[-1] if n_gauss else 0
        gauss_growth  = gauss_end - gauss_start
        gauss_pct     = (gauss_growth / max(gauss_start, 1)) * 100

        # Loss plateau detection: is the tail of the EMA flat?
        tail_len    = min(50, len(smoothed))
        ema_tail    = smoothed[-tail_len:]
        tail_range  = max(ema_tail) - min(ema_tail) if ema_tail else 0.0
        is_plateaued = tail_range < 0.001

        return {
            "iterations": {
                "first": iters[0] if iters else 0,
                "last":  iters[-1] if iters else 0,
                "count": len(iters),
            },
            "loss": {
                "initial":        round(losses[0], 6) if losses else None,
                "final":          round(losses[-1], 6) if losses else None,
                "min":            round(min(losses), 6) if losses else None,
                "improvement_pct": round((1 - losses[-1] / max(losses[0], 1e-8)) * 100, 1) if losses else 0,
                "ema_final":      round(smoothed[-1], 6) if smoothed else None,
                "convergence_rate_per_100": round(conv_rate * 100, 3),
                "is_plateaued":   is_plateaued,
            },
            "gaussians": {
                "start":      gauss_start,
                "end":        gauss_end,
                "growth":     gauss_growth,
                "growth_pct": round(gauss_pct, 1),
            },
            "vram": {
                "peak_gb":  round(max(v_peaks), 2) if v_peaks else 0.0,
                "final_gb": round(vrams[-1], 2) if vrams else 0.0,
                "mean_gb":  round(sum(vrams) / max(len(vrams), 1), 2),
            },
            "anomalies": {
                "nan_count":    last.get("nan_count", 0),
                "oom_count":    last.get("oom_count", 0),
                "loss_plateau": is_plateaued,
                "flat_gaussians": gauss_growth < gauss_start * 0.01,  # <1% growth
            },
        }

    def ascii_loss_curve(
        self,
        n_buckets: int = 50,
        height: int = 8,
        use_ema: bool = True,
    ) -> str:
        """Render an ASCII loss curve. Returns a multi-line string."""
        losses = self.losses()
        if not losses:
            return "  (no loss data)"

        series = ema(losses) if use_ema else losses

        bucket_size = max(1, len(series) // n_buckets)
        buckets = [series[i * bucket_size] for i in range(min(n_buckets, len(series)))]

        lo, hi = min(buckets), max(buckets)
        r = hi - lo if hi > lo else 1.0

        lines = []
        for row in range(height, 0, -1):
            threshold = lo + (row / height) * r
            bar = "".join("█" if v >= threshold else " " for v in buckets)
            y_label = f"{lo + (row / height) * r:.4f}"
            lines.append(f"  {y_label} │{bar}│")

        lines.append(f"  {'─' * (len(buckets) + 10)}")
        lines.append(f"  start{' ' * max(1, len(buckets) - 7)}end")
        return "\n".join(lines)

    def save_analytics(self, path: str) -> Path:
        """Write analytics.json alongside the metrics.json."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = self.summary_report()
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        tmp.rename(out)
        print(f"[analytics] Saved → {out}")
        return out
