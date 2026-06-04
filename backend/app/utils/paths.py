"""
backend/app/utils/paths.py
---------------------------
Shared path helpers for the MonoSplat backend.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root is 4 levels above this file: backend/app/utils/paths.py
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SRC_ROOT:  Path = REPO_ROOT / "src"
DATA_ROOT: Path = REPO_ROOT / "data"
EXPERIMENTS_ROOT: Path = REPO_ROOT / "experiments"
UPLOAD_ROOT: Path = DATA_ROOT / "uploads"


def ensure_dirs() -> None:
    """Create all required runtime directories if they do not exist."""
    for d in (DATA_ROOT, EXPERIMENTS_ROOT, UPLOAD_ROOT):
        d.mkdir(parents=True, exist_ok=True)


def experiment_dir(run_id: str) -> Path:
    return EXPERIMENTS_ROOT / run_id


def checkpoint_dir(run_id: str) -> Path:
    return experiment_dir(run_id) / "checkpoints"


def export_dir(run_id: str) -> Path:
    return experiment_dir(run_id) / "exports"


def upload_dir(project_id: str) -> Path:
    return UPLOAD_ROOT / project_id