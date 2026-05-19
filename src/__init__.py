"""
MonoSplat — Video to 3D Gaussian Splat pipeline.
"""

__version__ = "3.2.0"
__author__  = "MonoSplat"

# New in v3.2: hardened modules
from .utils.vram     import query_vram, VRAMStats, reset_peak_vram
from .utils.metrics  import PipelineMetrics, TrainingMetricsLog
from .external       import GSPLAT_AVAILABLE, GSPLAT_VERSION
from .diagnostics    import MetricsAnalyzer, validate_dataset

__all__ = [
    "query_vram", "VRAMStats", "reset_peak_vram",
    "PipelineMetrics", "TrainingMetricsLog",
    "GSPLAT_AVAILABLE", "GSPLAT_VERSION",
    "MetricsAnalyzer", "validate_dataset",
]
