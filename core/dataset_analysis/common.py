"""Shared helpers for dataset analysis modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(image_dir: str | Path) -> List[Path]:
    """Return top-level image files in stable order."""
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return float(sum(values) / len(values))


def sample_files(files: list[Path], max_samples: int = 60) -> list[Path]:
    """Evenly sample files for expensive diagnostics."""
    if len(files) <= max_samples:
        return files
    step = len(files) / max_samples
    return [files[int(i * step)] for i in range(max_samples)]
