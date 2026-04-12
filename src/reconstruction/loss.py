"""
loss.py
Training losses and evaluation metrics for Gaussian Splatting.

Additions to match LeoDarcy/360GS metrics.py
---------------------------------------------
- psnr_metric()  : peak signal-to-noise ratio (dB) — used in 360GS metrics.py
- ssim_metric()  : returns (1 - SSIM) for use as a loss; also callable as a metric
- lpips_metric() : LPIPS perceptual loss (optional — requires lpips package)
- combined_loss(): lambda_ssim controlled by config (360GS default: 0.2)
"""

import functools
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Kernel cache
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
# Loss functions
# ---------------------------------------------------------------------------

def l1_loss(rendered: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(rendered, target)


def ssim_metric(rendered: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """
    Returns (1 − SSIM) — lower is better. Used both as a loss and metric.
    To get raw SSIM score: 1.0 - ssim_metric(rendered, target).item()
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target   = target.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = rendered.shape[1]
    pad      = window_size // 2

    kernel = _get_ssim_kernel(window_size, channels, str(rendered.device))

    def conv(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=channels)

    mu1 = conv(rendered)
    mu2 = conv(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    s1_sq = conv(rendered * rendered) - mu1_sq
    s2_sq = conv(target   * target)   - mu2_sq
    s12   = conv(rendered * target)   - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))

    return 1.0 - ssim_map.mean()


# Keep original name as alias for backward compat
ssim_loss = ssim_metric


def psnr_metric(rendered: Tensor, target: Tensor) -> Tensor:
    """
    Peak Signal-to-Noise Ratio (dB).
    Matches 360GS metrics.py psnr() function.
    Higher is better. Typical range for Gaussian Splatting: 25–35 dB.
    """
    mse = F.mse_loss(rendered, target)
    mse = mse.clamp(min=1e-10)
    return 20.0 * torch.log10(torch.tensor(1.0, device=rendered.device)) - 10.0 * torch.log10(mse)


def lpips_metric(
    rendered: Tensor,
    target: Tensor,
    net: str = "vgg",
    _cache: dict = {},
) -> Tensor:
    """
    LPIPS perceptual loss. Matches 360GS lpipsPyTorch usage.
    Requires: pip install lpips

    Falls back to L1 if lpips is not installed (with a warning).
    The LPIPS model is cached after first call.
    """
    cache_key = (net, str(rendered.device))
    if cache_key not in _cache:
        try:
            import lpips
            _cache[cache_key] = lpips.LPIPS(net=net).to(rendered.device)
        except ImportError:
            print("[loss] WARNING: lpips not installed. Falling back to L1.")
            _cache[cache_key] = None

    lpips_fn = _cache[cache_key]
    if lpips_fn is None:
        return l1_loss(rendered, target)

    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target   = target.unsqueeze(0)

    # LPIPS expects images in [-1, 1]
    rendered_n = rendered * 2.0 - 1.0
    target_n   = target   * 2.0 - 1.0
    return lpips_fn(rendered_n, target_n).mean()


def combined_loss(
    rendered: Tensor,
    target: Tensor,
    lambda_ssim: float = 0.2,   # 360GS default
    lambda_lpips: float = 0.0,  # optional — set >0 in config to enable
) -> Tensor:
    """
    Combined loss: (1-λ)·L1 + λ·(1−SSIM) [+ lpips term if enabled].
    Matches 360GS training loss with lambda_dssim=0.2.
    """
    l1   = l1_loss(rendered, target)
    ssim = ssim_metric(rendered, target)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim
    if lambda_lpips > 0.0:
        loss = loss + lambda_lpips * lpips_metric(rendered, target)
    return loss