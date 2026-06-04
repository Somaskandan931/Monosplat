"""
core/evaluation/ssim.py
------------------------
Structural Similarity Index Measure (SSIM) evaluator for MonoSplat.

SSIM ∈ [0, 1] — higher is better.
Note: loss.ssim_metric() returns (1 - SSIM) for use as a training loss.
This module always returns raw SSIM scores (not inverted) for evaluation.

References
----------
  Wang et al. "Image Quality Assessment: From Error Visibility to Structural
  Similarity." IEEE TIP, 2004. doi:10.1109/TIP.2003.819861
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


ImageLike = Union[np.ndarray, str, Path]


class SSIMEvaluator:
    """
    Compute SSIM between rendered images and ground-truth frames.

    Parameters
    ----------
    window_size : int
        Size of the Gaussian kernel (default 11, standard in 3DGS literature).
    data_range  : float
        Maximum pixel value (1.0 for float images).
    """

    def __init__(self, window_size: int = 11, data_range: float = 1.0) -> None:
        self.window_size = window_size
        self.data_range  = data_range

    # ── Single-pair ────────────────────────────────────────────────────────

    def compute(self, rendered: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute SSIM for one (rendered, gt) image pair.

        Returns
        -------
        SSIM in [0, 1] (higher is better).
        """
        rendered     = _to_float32(rendered)
        ground_truth = _to_float32(ground_truth)

        if rendered.shape != ground_truth.shape:
            ground_truth = _resize_to(ground_truth, rendered.shape[:2])

        # Multi-channel: average SSIM across channels
        if rendered.ndim == 3:
            ssim_vals = [
                _ssim_channel(rendered[..., c], ground_truth[..., c],
                              self.window_size, self.data_range)
                for c in range(rendered.shape[2])
            ]
            return float(np.mean(ssim_vals))
        return float(_ssim_channel(rendered, ground_truth, self.window_size, self.data_range))

    # ── Dataset-level ──────────────────────────────────────────────────────

    def evaluate_set(
        self,
        rendered_images: Sequence[ImageLike],
        gt_images: Sequence[ImageLike],
        names: Optional[Sequence[str]] = None,
    ) -> Dict:
        """
        Evaluate SSIM over a set of (rendered, gt) image pairs.

        Returns
        -------
        dict with keys: mean, min, max, std, per_image, count
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
            val = self.compute(_load(rend), _load(gt))
            per_image.append({"name": name, "ssim": round(val, 4)})
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

    # ── Torch-native path ──────────────────────────────────────────────────

    @staticmethod
    def from_tensors(rendered, ground_truth) -> float:
        """
        Compute raw SSIM (not inverted) from PyTorch tensors (C,H,W) in [0,1].
        """
        import sys
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[2]
        for _p in (str(_repo / "src"), str(_repo)):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from reconstruction.loss import ssim_metric
        # ssim_metric returns (1 - SSIM), so invert
        val = 1.0 - ssim_metric(rendered, ground_truth)
        return float(val.detach().cpu().item())


# ── Pure-numpy SSIM kernel ────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - size // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _gaussian_kernel_2d(size: int, sigma: float = 1.5) -> np.ndarray:
    k1d = _gaussian_kernel_1d(size, sigma)
    return np.outer(k1d, k1d)


def _ssim_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int,
    data_range: float,
) -> float:
    """Compute SSIM for a single-channel (H, W) float32 pair."""
    from scipy.ndimage import convolve

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_2d(window_size)

    def filt(x):
        return convolve(x.astype(np.float64), kernel, mode="reflect")

    mu1 = filt(img1)
    mu2 = filt(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    s1_sq = filt(img1 * img1) - mu1_sq
    s2_sq = filt(img2 * img2) - mu2_sq
    s12   = filt(img1 * img2) - mu12

    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2)

    ssim_map = np.where(den > 0, num / den, 0.0)
    return float(ssim_map.mean())


# ── Image helpers ──────────────────────────────────────────────────────────────

def _load(img: ImageLike) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img.astype(np.float32)
    from PIL import Image
    arr = np.array(Image.open(str(img)).convert("RGB"), dtype=np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


def _to_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return (arr / 255.0 if arr.max() > 1.0 + 1e-6 else arr).clip(0.0, 1.0)


def _resize_to(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = pil.resize((hw[1], hw[0]), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0
