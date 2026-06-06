"""
loss.py — Training losses and evaluation metrics for MonoSplat.

Based on the graphdeco-inria/gaussian-splatting loss functions, extended
with LPIPS and PSNR metrics matching 360GS / Scaffold-GS conventions.
"""

import functools
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# SSIM kernel cache
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _get_ssim_kernel(window_size: int, channels: int, device_str: str) -> Tensor:
    coords   = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel1  = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel1  = kernel1 / kernel1.sum()
    kernel2d = kernel1.unsqueeze(0) * kernel1.unsqueeze(1)
    kernel   = kernel2d.unsqueeze(0).unsqueeze(0)
    kernel   = kernel.expand(channels, 1, window_size, window_size)
    return kernel.to(device_str)


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def l1_loss(rendered: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(rendered, target)


def ssim_metric(rendered: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """
    Returns (1 − SSIM) — lower is better.
    To get raw SSIM score: 1.0 - ssim_metric(rendered, target).item()
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target   = target.unsqueeze(0)

    C1       = 0.01 ** 2
    C2       = 0.03 ** 2
    channels = rendered.shape[1]
    pad      = window_size // 2
    kernel   = _get_ssim_kernel(window_size, channels, str(rendered.device))

    def conv(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=channels)

    mu1    = conv(rendered)
    mu2    = conv(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2
    s1_sq  = conv(rendered * rendered) - mu1_sq
    s2_sq  = conv(target   * target)   - mu2_sq
    s12    = conv(rendered * target)   - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))

    return 1.0 - ssim_map.mean()


# Alias for backwards compatibility
ssim_loss = ssim_metric


def psnr_metric(rendered: Tensor, target: Tensor) -> Tensor:
    """
    Peak Signal-to-Noise Ratio (dB). Higher is better.
    Typical range for Gaussian Splatting on T&T: 23–28 dB.
    """
    mse = F.mse_loss(rendered, target).clamp(min=1e-10)
    return -10.0 * torch.log10(mse)


_lpips_cache: dict = {}


def lpips_metric(
    rendered: Tensor,
    target:   Tensor,
    net:      str = "vgg",
) -> Tensor:
    """
    LPIPS perceptual loss (requires: pip install lpips).
    Falls back to L1 if lpips is not installed.
    Model is cached after first call.
    """
    cache_key = (net, str(rendered.device))
    if cache_key not in _lpips_cache:
        try:
            import lpips
            _lpips_cache[cache_key] = lpips.LPIPS(net=net).to(rendered.device)
        except ImportError:
            print("[loss] WARNING: lpips not installed. Falling back to L1.")
            _lpips_cache[cache_key] = None

    lpips_fn = _lpips_cache[cache_key]
    if lpips_fn is None:
        return l1_loss(rendered, target)

    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target   = target.unsqueeze(0)

    # LPIPS expects images in [-1, 1]
    return lpips_fn(rendered * 2.0 - 1.0, target * 2.0 - 1.0).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def combined_loss(
    rendered:     Tensor,
    target:       Tensor,
    lambda_ssim:  float = 0.2,
    lambda_lpips: float = 0.1,
) -> Tensor:
    """
    Combined loss: (1 - λ_ssim - λ_lpips)·L1 + λ_ssim·(1−SSIM) + λ_lpips·LPIPS.

    Weights sum to 1.0 — LPIPS weight is absorbed from L1, not added on top.
    Callers should always pass lambda values from the loaded config explicitly.
    """
    l1_weight = max(0.0, 1.0 - lambda_ssim - lambda_lpips)
    loss      = l1_weight * l1_loss(rendered, target) \
              + lambda_ssim * ssim_metric(rendered, target)

    if lambda_lpips > 0.0:
        loss = loss + lambda_lpips * lpips_metric(rendered, target)

    return loss