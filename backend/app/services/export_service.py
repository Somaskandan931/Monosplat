"""
backend/app/services/export_service.py
----------------------------------------
ExportService — delegates to colab/export_splat.py.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("monosplat.services.export")

_REPO_ROOT     = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = _REPO_ROOT / "colab" / "export_splat.py"


class ExportService:
    """Export a trained Gaussian model checkpoint to PLY / SPLAT formats."""

    def run(
        self,
        checkpoint_path: str,
        output_dir: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Export checkpoint to the requested formats.

        Parameters
        ----------
        checkpoint_path : path to .ckpt file
        output_dir      : directory to write output files
        formats         : list of formats, e.g. ["ply", "splat"] (default: both)

        Returns
        -------
        dict mapping format → output file path
        """
        if not _EXPORT_SCRIPT.exists():
            raise FileNotFoundError(
                f"export_splat.py not found at {_EXPORT_SCRIPT}."
            )

        formats = formats or ["ply", "splat"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(_EXPORT_SCRIPT),
            "--checkpoint", checkpoint_path,
            "--output",     output_dir,
        ]

        log.info("Exporting: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Export subprocess exited with code {result.returncode}"
            )

        out_dir = Path(output_dir)
        outputs = {}
        for fmt in formats:
            candidates = list(out_dir.glob(f"*.{fmt}"))
            if candidates:
                outputs[fmt] = str(candidates[-1])

        return outputs
