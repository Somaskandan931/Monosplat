"""
backend/app/services/result_service.py
----------------------------------------
Handles importing Colab training results (PLY / SPLAT) back into the job.

Flow:
  Colab trains → exports final.ply + final.splat → user uploads results.zip
  → result_service unpacks → marks job complete with viewer paths
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

log = logging.getLogger("monosplat.services.result")

_RESULTS_ROOT = Path("data/results")


def import_results(zip_path: str, job_id: str) -> dict:
    """
    Unpack a Colab results ZIP and return paths to viewer-ready files.

    Expected ZIP contents (minimum):
        exports/final.ply
        exports/final.splat   (optional)

    Returns
    -------
    dict:
        output_dir   — directory where files were extracted
        ply_path     — path to final.ply (or None)
        splat_path   — path to final.splat (or None)
    """
    output_dir = _RESULTS_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(output_dir))

    ply   = output_dir / "exports" / "final.ply"
    splat = output_dir / "exports" / "final.splat"

    # Fallback: search recursively if not in expected location
    if not ply.exists():
        found = list(output_dir.rglob("*.ply"))
        ply = found[-1] if found else None

    if not splat.exists():
        found = list(output_dir.rglob("*.splat"))
        splat = found[-1] if found else None

    log.info(
        "Results imported to %s — ply=%s splat=%s",
        output_dir,
        ply,
        splat,
    )

    return {
        "output_dir": str(output_dir),
        "ply_path":   str(ply)   if ply   else None,
        "splat_path": str(splat) if splat else None,
    }
