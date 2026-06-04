"""
core/evaluation/psnr.py
------------------------
Peak Signal-to-Noise Ratio evaluator for MonoSplat.

Computes PSNR over a set of (rendered, ground-truth) image pairs,
returning per-image values and an aggregate summary.

References
----------
  - Original 3DGS paper uses PSNR as the primary quality metric.
  - Formula: PSNR = -10 · log₁₀(MSE), where images are in [0, 1].

Design notes
------------
  - Delegates pixel-level math to src/reconstruction/loss.psnr_metric
    so there is exactly one PSNR implementation in the codebase.
  - Also provides a lightweight numpy fallback for evaluation without
    a GPU / PyTorch tensors (e.g. loading saved PNG test renders).
  - All results are plain Python floats for JSON serialisation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ── Types ─────────────────────────────────────────────────────────────────────
# Images can be supplied as numpy arrays (H,W,C) float32 in [0,1]
# or as file paths pointing to PNG/JPEG renders.
ImageLike = Union[np.ndarray, str, Path]


# ── Public API ────────────────────────────────────────────────────────────────

class PSNREvaluator:
    """
    Compute PSNR between rendered images and ground-truth frames.

    Parameters
    ----------
    data_range : float
        Maximum possible pixel value (default 1.0 for float images).
    """

    def __init__(self, data_range: float = 1.0) -> None:
        self.data_range = data_range

    # ── Single-pair ────────────────────────────────────────────────────────

    def compute(self, rendered: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute PSNR between one rendered image and its ground truth.

        Parameters
        ----------
        rendered, ground_truth : float32 numpy arrays (H, W, C) in [0, 1]

        Returns
        -------
        PSNR in dB (higher is better). Returns 0.0 on degenerate input.
        """
        rendered     = _to_float32(rendered)
        ground_truth = _to_float32(ground_truth)

        if rendered.shape != ground_truth.shape:
            ground_truth = _resize_to(ground_truth, rendered.shape[:2])

        mse = float(np.mean((rendered - ground_truth) ** 2))
        if mse < 1e-12:
            return float("inf")
        return float(10.0 * math.log10((self.data_range ** 2) / mse))

    # ── Dataset-level ──────────────────────────────────────────────────────

    def evaluate_set(
        self,
        rendered_images: Sequence[ImageLike],
        gt_images: Sequence[ImageLike],
        names: Optional[Sequence[str]] = None,
    ) -> Dict:
        """
        Evaluate PSNR over a set of (rendered, gt) image pairs.

        Returns
        -------
        dict with keys:
            mean, min, max, std           — aggregate statistics (dB)
            per_image                     — list of {name, psnr} dicts
            count                         — number of image pairs evaluated
        """
        if len(rendered_images) != len(gt_images):
            raise ValueError(
                f"Rendered/GT count mismatch: {len(rendered_images)} vs {len(gt_images)}"
            )
        if names is None:
            names = [f"frame_{i:04d}" for i in range(len(rendered_images))]

        per_image: List[Dict] = []
        values: List[float]   = []

        for name, rend, gt in zip(names, rendered_images, gt_images):
            rend_arr = _load(rend)
            gt_arr   = _load(gt)
            val      = self.compute(rend_arr, gt_arr)
            per_image.append({"name": name, "psnr": round(val, 4)})
            if math.isfinite(val):
                values.append(val)

        if not values:
            return {"mean": None, "min": None, "max": None, "std": None,
                    "per_image": per_image, "count": 0}

        arr = np.array(values, dtype=np.float64)
        return {
            "mean":      round(float(arr.mean()), 4),
            "min":       round(float(arr.min()),  4),
            "max":       round(float(arr.max()),  4),
            "std":       round(float(arr.std()),  4),
            "per_image": per_image,
            "count":     len(values),
        }

    # ── Torch-native path (used during training for speed) ─────────────────

    @staticmethod
    def from_tensors(rendered, ground_truth) -> float:
        """
        Compute PSNR from PyTorch tensors via the existing loss module.
        Both tensors must be float32, shape (C, H, W), in [0, 1].

        Returns float (dB).
        """
        import sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[2]
        for _p in (str(_repo / "src"), str(_repo)):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from reconstruction.loss import psnr_metric
        val = psnr_metric(rendered, ground_truth)
        return float(val.detach().cpu().item())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(img: ImageLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img.astype(np.float32)
    from PIL import Image
    arr = np.array(Image.open(str(img)).convert("RGB"), dtype=np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


def _to_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return arr.clip(0.0, 1.0)


def _resize_to(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0
