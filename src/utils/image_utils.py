"""
src/utils/image_utils.py
-------------------------
Image loading, resizing, and conversion utilities.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple


def load_image_rgb(path: str, width: int = 0, height: int = 0) -> np.ndarray:
    """
    Load a JPEG/PNG as an RGB float32 array normalized to [0, 1].

    Args:
        path:   Image file path.
        width:  Target width  (0 = keep original).
        height: Target height (0 = keep original).

    Returns:
        np.ndarray of shape (H, W, 3), dtype float32, values in [0, 1].
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if requested
    if width > 0 and height > 0:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32) / 255.0


def image_to_tensor(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert (H, W, 3) float32 numpy array → (1, 3, H, W) torch Tensor.

    Args:
        img:    Input image array.
        device: Target device.

    Returns:
        Tensor of shape (1, 3, H, W).
    """
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert (1, 3, H, W) or (3, H, W) tensor → (H, W, 3) uint8 numpy array.

    Args:
        tensor: Input tensor.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dim

    img = tensor.detach().cpu().float().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def save_image(img: np.ndarray, path: str):
    """
    Save a uint8 RGB numpy array to disk as JPEG.

    Args:
        img:  Image array (H, W, 3), uint8.
        path: Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Higher is better. Typical range: 20–40 dB.

    Args:
        pred: Predicted image tensor.
        gt:   Ground truth image tensor.

    Returns:
        PSNR value in dB.
    """
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return float(-10 * torch.log10(mse))
