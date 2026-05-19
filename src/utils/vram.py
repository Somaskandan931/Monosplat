"""
vram.py
Correct VRAM telemetry for CUDA training.

Design rules
------------
- Always call torch.cuda.synchronize() before querying memory.
- Return allocated, reserved, and peak VRAM in GB.
- Gracefully returns zeros on CPU-only machines.
- Thread-safe: each call is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VRAMStats:
    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    peak_gb: float = 0.0
    free_gb: float = 0.0

    def as_dict(self) -> dict:
        return {
            "vram_alloc_gb": round(self.allocated_gb, 2),
            "vram_resv_gb":  round(self.reserved_gb,  2),
            "vram_peak_gb":  round(self.peak_gb,       2),
            "vram_free_gb":  round(self.free_gb,       2),
        }

    def pbar_str(self) -> str:
        """Compact string for tqdm postfix."""
        return f"{self.allocated_gb:.2f}G/{self.peak_gb:.2f}G"

    def __str__(self) -> str:
        return (
            f"alloc={self.allocated_gb:.2f}GB "
            f"resv={self.reserved_gb:.2f}GB "
            f"peak={self.peak_gb:.2f}GB "
            f"free={self.free_gb:.2f}GB"
        )


def query_vram(device: str = "cuda") -> VRAMStats:
    """
    Query current VRAM usage.

    Calls torch.cuda.synchronize() first to ensure the device has finished
    all pending operations before the memory counters are read.
    Returns zeros silently on CPU-only machines.
    """
    if not torch.cuda.is_available():
        return VRAMStats()

    try:
        torch.cuda.synchronize(device=device)
        allocated = torch.cuda.memory_allocated(device=device)
        reserved  = torch.cuda.memory_reserved(device=device)
        peak      = torch.cuda.max_memory_allocated(device=device)
        free_b, _ = torch.cuda.mem_get_info(device=device)

        return VRAMStats(
            allocated_gb=allocated / 1e9,
            reserved_gb=reserved   / 1e9,
            peak_gb=peak           / 1e9,
            free_gb=free_b         / 1e9,
        )
    except Exception:
        # Non-fatal: return zeros rather than crashing training
        return VRAMStats()


def reset_peak_vram(device: str = "cuda") -> None:
    """Reset the peak VRAM counter (call at training start)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)


def total_vram_gb(device: str = "cuda") -> float:
    """Total VRAM on the device in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 1e9
