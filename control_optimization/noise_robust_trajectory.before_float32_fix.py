"""Compatibility import for the canonical trajectory implementation."""
from .trajectory import TrajectoryMetrics, propagate_with_electron_metrics

__all__ = ["TrajectoryMetrics", "propagate_with_electron_metrics"]
