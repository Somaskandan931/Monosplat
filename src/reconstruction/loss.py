"""
loss.py
Training losses for Gaussian Splatting: L1, SSIM, and combined loss.

Performance improvements:
- SSIM Gaussian kernel is cached per (window_size, channels, device) so it is
  not rebuilt every iteration (previously re-built every forward pass).
- Combined loss uses a single function call with minimal overhead.
"""

import functools

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Kernel cache: avoids recomputing the SSIM window every training step
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _get_ssim_kernel(window_size: int, channels: int, device_str: str) -> Tensor:
    """Build and cache a (channels, 1, W, W) Gaussian kernel for SSIM."""
    coords  = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel1 = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel1 = kernel1 / kernel1.sum()
    kernel2d = kernel1.unsqueeze(0) * kernel1.unsqueeze(1)           # (W, W)
    kernel   = kernel2d.unsqueeze(0).unsqueeze(0)                    # (1, 1, W, W)
    kernel   = kernel.expand(channels, 1, window_size, window_size)  # (C, 1, W, W)
    return kernel.to(device_str)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def l1_loss(rendered: Tensor, target: Tensor) -> Tensor:
    """Pixel-wise L1 loss."""
    return F.l1_loss(rendered, target)


def ssim_loss(rendered: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """
    Structural Similarity (SSIM) loss.
    Returns (1 − SSIM) so lower is better, consistent with gradient descent.

    Args:
        rendered:    (B, C, H, W) or (C, H, W) predicted image in [0, 1].
        target:      Same shape as *rendered*.
        window_size: Size of the Gaussian smoothing window (default 11).

    Returns:
        Scalar loss tensor.
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        target   = target.unsqueeze(0)

    C1       = 0.01 ** 2
    C2       = 0.03 ** 2
    channels = rendered.shape[1]
    pad      = window_size // 2

    # Cached kernel — no allocation after the first call for this config
    kernel = _get_ssim_kernel(window_size, channels, str(rendered.device))

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


def combined_loss(
    rendered: Tensor,
    target: Tensor,
    lambda_ssim: float = 0.2,
) -> Tensor:
    """
    Combined L1 + SSIM loss (3DGS paper default).

    Loss = (1 − λ) · L1 + λ · (1 − SSIM)

    Args:
        rendered:    Predicted image tensor.
        target:      Ground-truth image tensor.
        lambda_ssim: SSIM weight (paper default 0.2).

    Returns:
        Scalar loss tensor.
    """
    l1   = l1_loss(rendered, target)
    ssim = ssim_loss(rendered, target)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim