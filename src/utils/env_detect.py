"""
env_detect.py
Auto-detect the compute environment and choose the right settings.

Usage
-----
    from src.utils.env_detect import should_use_gpu, is_colab, get_env_info
"""

import os
import subprocess
import sys


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "COLAB_GPU" in os.environ or "COLAB_BACKEND_URL" in os.environ


def is_jupyter() -> bool:
    """Return True when running inside any Jupyter notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def has_cuda_colmap() -> bool:
    """
    Return True if the installed COLMAP binary was built with CUDA support.
    Checks for CUDA mention in colmap --help output.
    """
    try:
        result = subprocess.run(
            ["colmap", "--help"],
            capture_output=True, text=True, timeout=10
        )
        output = (result.stdout + result.stderr).lower()
        return "cuda" in output or "gpu" in output
    except Exception:
        return False


def has_opengl_context() -> bool:
    """
    Return True if an OpenGL context is likely available.
    Headless servers (Colab, Docker, SSH without X11) typically do NOT.
    """
    # If DISPLAY is not set and we're on Linux, no GUI → no OpenGL context
    if sys.platform.startswith("linux"):
        display = os.environ.get("DISPLAY", "")
        wayland = os.environ.get("WAYLAND_DISPLAY", "")
        if not display and not wayland:
            return False
    return True


def has_torch_gpu() -> bool:
    """Return True if PyTorch can see at least one CUDA device."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def should_use_gpu() -> bool:
    """
    Master switch: should COLMAP SiftExtraction use GPU?

    Rules (in priority order):
      1. If we're in Colab                          → CPU (OpenGL unstable)
      2. If no OpenGL context detected              → CPU
      3. If COLMAP wasn't built with CUDA           → CPU
      4. Otherwise                                  → GPU
    """
    if is_colab():
        return False
    if not has_opengl_context():
        return False
    if not has_cuda_colmap():
        return False
    return True


def should_use_matching_gpu() -> bool:
    """
    Feature *matching* is safe on GPU even in headless environments
    because it doesn't require an OpenGL context.
    Returns True when a CUDA COLMAP build is present.
    """
    return has_cuda_colmap()


def get_env_info() -> dict:
    """Return a dict summarising the detected environment."""
    return {
        "is_colab":            is_colab(),
        "is_jupyter":          is_jupyter(),
        "has_opengl_context":  has_opengl_context(),
        "has_cuda_colmap":     has_cuda_colmap(),
        "has_torch_gpu":       has_torch_gpu(),
        "colmap_extraction_gpu": should_use_gpu(),
        "colmap_matching_gpu":   should_use_matching_gpu(),
    }


if __name__ == "__main__":
    import json
    info = get_env_info()
    print("[env_detect] Environment summary:")
    for k, v in info.items():
        status = "✓" if v else "✗"
        print(f"  {status}  {k}: {v}")