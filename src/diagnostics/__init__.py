"""src/diagnostics — training analytics and convergence diagnostics."""
from .analytics import MetricsAnalyzer, ema, convergence_rate
from .dataset_quality import validate as validate_dataset

__all__ = ["MetricsAnalyzer", "ema", "convergence_rate", "validate_dataset"]
