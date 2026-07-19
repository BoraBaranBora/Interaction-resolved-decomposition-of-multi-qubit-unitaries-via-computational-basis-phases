"""Gradient-based control optimization for the NV tripartite gates."""

from .config import ControlConfig, WarmStartConfig, load_control_config
from .objective import SupportSelectiveObjective
from .pulse import (
    BoundedFourierPulse,
    DirectFourierPulse,
    FourierPulseBounds,
    ReferenceResidualPulse,
    WaveformFitResult,
)
from .runner import ControlOptimizer

__all__ = [
    "BoundedFourierPulse",
    "ControlConfig",
    "ControlOptimizer",
    "DirectFourierPulse",
    "FourierPulseBounds",
    "ReferenceResidualPulse",
    "SupportSelectiveObjective",
    "WarmStartConfig",
    "WaveformFitResult",
    "load_control_config",
]
