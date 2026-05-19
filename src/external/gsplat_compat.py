"""
gsplat_compat.py
Compatibility wrapper for gsplat — isolates API drift from the rest of the codebase.

Purpose
-------
- Pin-checks: warns early if gsplat version is outside the tested range.
- Wraps the rasterization call so the rest of the codebase doesn't
  depend on gsplat's exact keyword argument names.
- Provides a graceful UNAVAILABLE sentinel so callers can test
  `if gsplat_compat.AVAILABLE` without try/except everywhere.

Tested versions
---------------
- gsplat 1.4.x, 1.5.x  — fully supported
- gsplat 1.3.x          — may work, not tested (warning issued)
- gsplat 2.x+           — unknown API changes (warning issued)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Attempt import and version check
# ---------------------------------------------------------------------------

AVAILABLE: bool = False
VERSION: str = "unavailable"

_gsplat = None

try:
    import gsplat as _gsplat_module
    _gsplat = _gsplat_module
    VERSION = getattr(_gsplat, "__version__", "unknown")
    AVAILABLE = True
except ImportError:
    pass

# Version gates — update these as compatibility is validated
_MIN_MAJOR = 1
_MIN_MINOR = 4
_MAX_MAJOR = 1   # bump when gsplat 2.x API is audited

if AVAILABLE:
    try:
        parts = VERSION.split(".")
        major, minor = int(parts[0]), int(parts[1])
        if major < _MIN_MAJOR or (major == _MIN_MAJOR and minor < _MIN_MINOR):
            warnings.warn(
                f"[gsplat_compat] gsplat {VERSION} is older than the tested minimum "
                f"({_MIN_MAJOR}.{_MIN_MINOR}.x). Training may be unstable.",
                UserWarning, stacklevel=2,
            )
        elif major > _MAX_MAJOR:
            warnings.warn(
                f"[gsplat_compat] gsplat {VERSION} is newer than the tested maximum "
                f"({_MAX_MAJOR}.x.x). API may have changed — check for breaking changes.",
                UserWarning, stacklevel=2,
            )
    except (ValueError, IndexError):
        warnings.warn(
            f"[gsplat_compat] Cannot parse gsplat version '{VERSION}'. "
            "Proceeding without version validation.",
            UserWarning, stacklevel=2,
        )


def assert_compatible() -> None:
    """Raise ImportError if gsplat is not available."""
    if not AVAILABLE:
        raise ImportError(
            "gsplat is required for CUDA training but is not installed.\n"
            "Install with:  pip install gsplat\n"
            "Training will fall back to the software renderer (much slower)."
        )


# ---------------------------------------------------------------------------
# Wrapped rasterization call
# ---------------------------------------------------------------------------

def rasterize(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    backgrounds: Optional[torch.Tensor] = None,
    packed: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Thin wrapper around gsplat.rasterization().

    Returns
    -------
    render_colors : Tensor  (B, H, W, 3)
    render_alphas : Tensor  (B, H, W, 1)
    meta          : dict    contains means2d, radii, gaussian_ids for densification
    """
    assert_compatible()

    render_colors, render_alphas, meta = _gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=0,          # colours already evaluated before this call
        near_plane=near_plane,
        far_plane=far_plane,
        backgrounds=backgrounds,
        packed=packed,
        render_mode="RGB",
    )
    return render_colors, render_alphas, meta


def check_meta_keys(meta: dict) -> None:
    """
    Warn if expected meta keys are absent (API drift detection).
    Called once at the start of training to catch version breakage early.
    """
    expected = {"means2d", "radii"}
    missing = expected - set(meta.keys())
    if missing:
        warnings.warn(
            f"[gsplat_compat] Expected meta keys missing after rasterization: {missing}.\n"
            "This may indicate an API change in gsplat. "
            "Densification will fall back to 3D position gradients (less accurate).",
            UserWarning, stacklevel=2,
        )
