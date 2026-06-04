"""
backend/app/services/training_service.py
-----------------------------------------
TrainingService — delegates to colab/train.py for GPU training.

For local GPU training this calls train.py as a subprocess so it runs in its
own process (GPU memory, clean imports). For Colab-based training, the caller
should package via run_pipeline() and upload the resulting ZIP instead.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("monosplat.services.training")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRAIN_SCRIPT = _REPO_ROOT / "colab" / "train.py"


class TrainingService:
    """
    Run MonoSplat Gaussian Splatting training.

    Delegates to colab/train.py via subprocess so GPU memory is isolated from
    the FastAPI process.
    """

    def run(
        self,
        sparse_path: str,
        image_dir: str,
        model_path: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        resume_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start training and block until completion.

        Parameters
        ----------
        sparse_path        : path to COLMAP sparse_text directory
        image_dir          : path to extracted frames directory
        model_path         : output directory for checkpoints
        config_overrides   : optional dict of config key=value overrides
        resume_checkpoint  : path to a checkpoint to resume from

        Returns
        -------
        dict with keys: checkpoint_path, model_path, status
        """
        if not _TRAIN_SCRIPT.exists():
            raise FileNotFoundError(
                f"train.py not found at {_TRAIN_SCRIPT}. "
                "Training must run in Google Colab — use run_pipeline() to "
                "package your dataset and upload the ZIP to Colab."
            )

        cmd = [
            sys.executable, str(_TRAIN_SCRIPT),
            "--sparse",  sparse_path,
            "--frames",  image_dir,
            "--output",  model_path,
        ]

        if resume_checkpoint:
            cmd += ["--resume", resume_checkpoint]

        if config_overrides:
            for key, value in config_overrides.items():
                cmd += [f"--{key}", str(value)]

        log.info("Starting training: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Training subprocess exited with code {result.returncode}"
            )

        # Find the latest checkpoint produced
        model_dir = Path(model_path)
        checkpoints = sorted(model_dir.rglob("checkpoint_*.ckpt"))
        latest_ckpt = str(checkpoints[-1]) if checkpoints else None

        return {
            "status":          "success",
            "model_path":      model_path,
            "checkpoint_path": latest_ckpt,
        }
