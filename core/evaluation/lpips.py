"""
core/evaluation/lpips.py
-------------------------
LPIPS (Learned Perceptual Image Patch Similarity) evaluator for MonoSplat.

LPIPS ∈ [0, 1] — lower is better (measures perceptual distance).
Uses the VGG backbone by default (same as 3DGS paper).

Graceful degradation: if the `lpips` package is not installed, falls back
to SSIM-based perceptual proxy with a clear warning — so evaluation still
runs on machines without the optional lpips dependency.

References
----------
  Zhang et al. "The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric." CVPR 2018. https://arxiv.org/abs/1801.03924
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("monosplat.evaluation.lpips")

ImageLike = Union[np.ndarray, str, Path]


class LPIPSEvaluator:
    """
    Compute LPIPS perceptual distance between rendered and ground-truth images.

    Parameters
    ----------
    net     : backbone network ('vgg', 'alex', 'squeeze')
    device  : 'cuda' / 'cpu' / None (auto-detect)
    """

    def __init__(self, net: str = "vgg", device: Optional[str] = None) -> None:
        self.net    = net
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._fn    = None          # lazy-loaded LPIPS model
        self._using_fallback = False

    # ── Single-pair ────────────────────────────────────────────────────────

    def compute(self, rendered: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute LPIPS for one (rendered, gt) image pair.

        Parameters
        ----------
        rendered, ground_truth : float32 numpy (H, W, C) in [0, 1]

        Returns
        -------
        LPIPS distance in [0, 1] (lower is better).
        """
        rendered     = _to_float32(rendered)
        ground_truth = _to_float32(ground_truth)
        if rendered.shape != ground_truth.shape:
            ground_truth = _resize_to(ground_truth, rendered.shape[:2])

        fn = self._get_fn()
        if self._using_fallback:
            return _ssim_proxy(rendered, ground_truth)

        import torch
        t_rend = _to_tensor(rendered, self.device)
        t_gt   = _to_tensor(ground_truth, self.device)
        with torch.no_grad():
            val = fn(t_rend, t_gt)
        return float(val.mean().item())

    # ── Dataset-level ──────────────────────────────────────────────────────

    def evaluate_set(
        self,
        rendered_images: Sequence[ImageLike],
        gt_images: Sequence[ImageLike],
        names: Optional[Sequence[str]] = None,
    ) -> Dict:
        """
        Evaluate LPIPS over a set of (rendered, gt) image pairs.

        Returns
        -------
        dict with: mean, min, max, std, per_image, count, using_fallback
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
            per_image.append({"name": name, "lpips": round(val, 4)})
            values.append(val)

        if not values:
            return {"mean": None, "min": None, "max": None, "std": None,
                    "per_image": per_image, "count": 0, "using_fallback": self._using_fallback}

        arr = np.array(values, dtype=np.float64)
        return {
            "mean":           round(float(arr.mean()), 4),
            "min":            round(float(arr.min()),  4),
            "max":            round(float(arr.max()),  4),
            "std":            round(float(arr.std()),  4),
            "per_image":      per_image,
            "count":          len(values),
            "using_fallback": self._using_fallback,
            "net":            self.net if not self._using_fallback else "ssim_proxy",
        }

    # ── Torch-native path ──────────────────────────────────────────────────

    @staticmethod
    def from_tensors(rendered, ground_truth, net: str = "vgg") -> float:
        """
        Compute LPIPS from PyTorch tensors (C, H, W) in [0, 1].
        Delegates to src/reconstruction/loss.lpips_metric.
        """
        _add_src_to_path()
        from reconstruction.loss import lpips_metric
        val = lpips_metric(rendered, ground_truth, net=net)
        return float(val.detach().cpu().item())

    # ── Lazy model loader ──────────────────────────────────────────────────

    def _get_fn(self):
        if self._fn is not None:
            return self._fn
        try:
            import torch
            import lpips as _lpips_pkg
            model = _lpips_pkg.LPIPS(net=self.net).to(self.device)
            model.eval()
            self._fn = model
            log.info("LPIPS model loaded (net=%s, device=%s)", self.net, self.device)
        except ImportError:
            warnings.warn(
                "[evaluation/lpips] 'lpips' package not installed. "
                "Falling back to SSIM-based proxy. "
                "Install with: pip install lpips",
                stacklevel=3,
            )
            self._using_fallback = True
            self._fn = None
        return self._fn


# ── Helpers ────────────────────────────────────────────────────────────────────

def _add_src_to_path() -> None:
    _repo = Path(__file__).resolve().parents[2]
    for _p in (str(_repo / "src"), str(_repo)):
        if _p not in sys.path:
            sys.path.insert(0, _p)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _to_tensor(arr: np.ndarray, device: str):
    """Convert (H, W, C) float32 numpy to (1, C, H, W) tensor in [-1, 1]."""
    import torch
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t * 2.0 - 1.0     # LPIPS expects [-1, 1]


def _ssim_proxy(rendered: np.ndarray, gt: np.ndarray) -> float:
    """Lightweight SSIM-based proxy for LPIPS when lpips package is absent."""
    from .ssim import SSIMEvaluator
    ssim = SSIMEvaluator().compute(rendered, gt)
    return round(1.0 - ssim, 4)   # invert: 0=identical, 1=very different


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
