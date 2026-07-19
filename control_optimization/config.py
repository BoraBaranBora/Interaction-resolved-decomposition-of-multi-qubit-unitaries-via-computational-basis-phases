from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WarmStartConfig:
    """Settings for converting a saved waveform to the selected pulse ansatz."""

    enabled: bool = True
    fit_steps: int = 500
    fit_restarts: int = 1
    fit_lbfgs_steps: int = 0
    learning_rate: float = 5.0e-2
    gradient_clip_norm: float = 10.0
    warning_relative_error: float = 5.0e-2
    evaluate_reference: bool = True
    accept_imperfect_fit: bool = False
    minimum_corrected_fidelity: float = 0.0
    cache_fit: bool = True


@dataclass(frozen=True)
class AdamConfig:
    enabled: bool = True
    steps: int = 100
    learning_rate: float = 2.0e-2
    steps_per_ns: float = 0.25
    gradient_clip_norm: float = 10.0


@dataclass(frozen=True)
class LBFGSConfig:
    enabled: bool = True
    max_iter: int = 50
    history_size: int = 20
    tolerance_grad: float = 1.0e-9
    tolerance_change: float = 1.0e-12
    steps_per_ns: float = 0.5
    line_search_fn: str = "strong_wolfe"


@dataclass(frozen=True)
class ObjectiveWeights:
    corrected_infidelity: float = 1.0
    selected_phase: float = 1.0
    diagonality: float = 1.0
    unitarity: float = 0.2
    survival: float = 0.2
    peak: float = 0.0
    fluence: float = 1.0e-5
    smoothness: float = 1.0e-4
    electron_dephasing_exposure: float = 0.0


@dataclass(frozen=True)
class ControlConfig:
    gate: str
    duration_ns: float
    basis_size: int
    output_dir: str
    resume_from: str | None = None
    pulse_parameterization: str = "reference_residual_fourier"
    seed: int = 7
    active_carbons: tuple[int, ...] = (1, 2)
    logical_carbons: tuple[int, int] = (1, 2)
    mI_block: int = 0
    electron_map: tuple[str, str] = ("m1", "0")
    target_angle_rad: float = 0.7853981633974483
    max_rabi_mhz: float = 5.0
    min_frequency_mhz: float = -5.0
    max_frequency_mhz: float = 5.0
    phase_bound_pi: float = 1.0
    taper_fraction: float = 0.15
    pair_weight: float = 0.2
    tripartite_weight: float = 0.5
    final_steps_per_ns: float = 1.0
    trajectory_sample_stride: int = 1
    validate_trajectory_propagator: bool = True
    deterministic_algorithms: bool = False
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    adam: AdamConfig = field(default_factory=AdamConfig)
    lbfgs: LBFGSConfig = field(default_factory=LBFGSConfig)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    def validate(self) -> None:
        gate = self.gate.lower()
        if gate not in {"zzz", "xzz"}:
            raise ValueError("gate must be 'zzz' or 'xzz'.")
        if self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive.")
        if self.basis_size < 1:
            raise ValueError("basis_size must be positive.")
        parameterization = self.pulse_parameterization.lower()
        if parameterization not in {
            "reference_residual_fourier",
            "direct_fourier",
            "bounded_fourier",
        }:
            raise ValueError(
                "pulse_parameterization must be 'direct_fourier', "
                "'reference_residual_fourier', or 'bounded_fourier'."
            )
        if not 0.0 < self.taper_fraction <= 0.5:
            raise ValueError("taper_fraction must lie in (0, 0.5].")
        if self.min_frequency_mhz >= self.max_frequency_mhz:
            raise ValueError("min_frequency_mhz must be below max_frequency_mhz.")
        if self.max_rabi_mhz <= 0:
            raise ValueError("max_rabi_mhz must be positive.")
        if self.final_steps_per_ns <= 0:
            raise ValueError("final_steps_per_ns must be positive.")
        if self.trajectory_sample_stride < 1:
            raise ValueError("trajectory_sample_stride must be positive.")
        if self.objective_weights.electron_dephasing_exposure < 0.0:
            raise ValueError("objective_weights.electron_dephasing_exposure cannot be negative.")
        if len(self.logical_carbons) != 2:
            raise ValueError("logical_carbons must contain exactly two carbon labels.")
        if not set(self.logical_carbons).issubset(set(self.active_carbons)):
            raise ValueError("logical_carbons must be present in active_carbons.")
        if self.electron_map not in {("m1", "0"), ("0", "m1")}:
            raise ValueError("electron_map must be ('m1','0') or ('0','m1').")
        if self.warm_start.fit_steps < 0:
            raise ValueError("warm_start.fit_steps cannot be negative.")
        if self.warm_start.fit_restarts < 1:
            raise ValueError("warm_start.fit_restarts must be at least one.")
        if self.warm_start.fit_lbfgs_steps < 0:
            raise ValueError("warm_start.fit_lbfgs_steps cannot be negative.")
        if not 0.0 <= self.warm_start.minimum_corrected_fidelity <= 1.0:
            raise ValueError(
                "warm_start.minimum_corrected_fidelity must lie in [0, 1]."
            )
        for name, value in {
            "adam.steps_per_ns": self.adam.steps_per_ns,
            "lbfgs.steps_per_ns": self.lbfgs.steps_per_ns,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list) else tuple(value)


def load_control_config(path: str | Path) -> ControlConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    raw = dict(raw)
    raw["active_carbons"] = _tuple(raw.get("active_carbons", (1, 2)))
    raw["logical_carbons"] = _tuple(raw.get("logical_carbons", (1, 2)))
    raw["electron_map"] = _tuple(raw.get("electron_map", ("m1", "0")))
    raw["warm_start"] = WarmStartConfig(**raw.get("warm_start", {}))
    raw["adam"] = AdamConfig(**raw.get("adam", {}))
    raw["lbfgs"] = LBFGSConfig(**raw.get("lbfgs", {}))
    raw["objective_weights"] = ObjectiveWeights(**raw.get("objective_weights", {}))

    config = ControlConfig(**raw)
    config.validate()
    return config
