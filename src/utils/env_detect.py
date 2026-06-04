"""
src/utils/env_detect.py
-----------------------
Environment detection helpers for COLMAP GPU and runtime context.

Used by colmap_runner.py to decide whether to enable GPU flags for
feature extraction and feature matching without crashing on CPU-only
machines or COLMAP builds without CUDA support.
"""

import shutil
import subprocess
import sys
from functools import lru_cache


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return "google.colab" in sys.modules


@lru_cache(maxsize=1)
def has_cuda_colmap(colmap_binary: str = "colmap") -> bool:
    """
    Return True if this COLMAP build was compiled with CUDA support.

    Probes by running `colmap feature_extractor --help` and checking
    whether the output mentions GPU/CUDA options. Result is cached so
    the probe only runs once per process.
    """
    if shutil.which(colmap_binary) is None:
        return False
    try:
        result = subprocess.run(
            [colmap_binary, "feature_extractor", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = (result.stdout + result.stderr).lower()
        return "use_gpu" in combined
    except Exception:
        return False


def should_use_gpu(colmap_binary: str = "colmap") -> bool:
    """
    Return True if GPU feature extraction should be attempted.

    Requires:
      - COLMAP built with CUDA  (has_cuda_colmap)
      - A CUDA GPU is visible to PyTorch
    """
    if not has_cuda_colmap(colmap_binary):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def should_use_matching_gpu(colmap_binary: str = "colmap") -> bool:
    """
    Return True if GPU feature *matching* should be attempted.

    Same requirements as extraction. Matching GPU flag is probed
    separately in colmap_runner._probe_matching_gpu_flag().
    """
    return should_use_gpu(colmap_binary)


def has_torch_gpu() -> bool:
    """Return True if PyTorch can see at least one CUDA GPU."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_env_info() -> dict:
    """
    Return a summary dict of the current runtime environment.
    Useful for pipeline logs and diagnostics.
    """
    import platform

    gpu_name = None
    vram_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass

    return {
        "platform":        platform.system(),
        "python":          sys.version.split()[0],
        "in_colab":        is_colab(),
        "has_torch_gpu":   has_torch_gpu(),
        "gpu_name":        gpu_name,
        "vram_gb":         vram_gb,
        "has_cuda_colmap": has_cuda_colmap(),
    }
