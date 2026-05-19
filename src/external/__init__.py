"""src/external — wrappers for third-party dependencies."""
from .gsplat_compat import AVAILABLE as GSPLAT_AVAILABLE, VERSION as GSPLAT_VERSION

__all__ = ["GSPLAT_AVAILABLE", "GSPLAT_VERSION"]
