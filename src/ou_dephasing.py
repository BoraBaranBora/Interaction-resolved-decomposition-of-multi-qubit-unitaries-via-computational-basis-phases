"""Ornstein--Uhlenbeck dephasing utilities for NV-register simulations.

The stochastic detuning beta(t) is expressed in angular-frequency units (rad/s)
and obeys

    d beta = -(beta - mu) dt / tau_c + sqrt(2 sigma^2 / tau_c) dW.

The exact discrete-time update is used, so the stationary variance and
correlation time are preserved for arbitrary simulation step size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import math
import numpy as np
import torch


@dataclass(frozen=True)
class OUParameters:
    """Parameters of a stationary Ornstein--Uhlenbeck frequency process.

    Attributes
    ----------
    correlation_time_s:
        Correlation time tau_c in seconds.
    sigma_rad_s:
        Stationary standard deviation sigma of beta(t), in rad/s.
    mean_rad_s:
        Stationary mean of beta(t), in rad/s. Pure dephasing normally uses zero.
    calibration:
        Human-readable record of how sigma was selected.
    coherence_time_s:
        Optional Ramsey T2* or Hahn-echo T2 used for calibration.
    """

    correlation_time_s: float
    sigma_rad_s: float
    mean_rad_s: float = 0.0
    calibration: str = "direct"
    coherence_time_s: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.correlation_time_s) or self.correlation_time_s <= 0:
            raise ValueError("correlation_time_s must be finite and positive")
        if not math.isfinite(self.sigma_rad_s) or self.sigma_rad_s < 0:
            raise ValueError("sigma_rad_s must be finite and non-negative")
        if not math.isfinite(self.mean_rad_s):
            raise ValueError("mean_rad_s must be finite")
        if self.coherence_time_s is not None:
            if not math.isfinite(self.coherence_time_s) or self.coherence_time_s <= 0:
                raise ValueError("coherence_time_s must be finite and positive")

    @classmethod
    def from_t2_star(
        cls,
        t2_star_s: float,
        correlation_time_s: float,
        *,
        mean_rad_s: float = 0.0,
    ) -> "OUParameters":
        """Choose sigma so Ramsey coherence is exp(-1) at t = T2*."""
        x = float(t2_star_s) / float(correlation_time_s)
        shape = x - 1.0 + math.exp(-x)
        if shape <= 0:
            raise ValueError("Invalid Ramsey calibration parameters")
        sigma = 1.0 / (float(correlation_time_s) * math.sqrt(shape))
        return cls(
            correlation_time_s=float(correlation_time_s),
            sigma_rad_s=sigma,
            mean_rad_s=float(mean_rad_s),
            calibration="ramsey_t2_star",
            coherence_time_s=float(t2_star_s),
        )

    @classmethod
    def from_hahn_echo_t2(
        cls,
        t2_echo_s: float,
        correlation_time_s: float,
        *,
        mean_rad_s: float = 0.0,
    ) -> "OUParameters":
        """Choose sigma so Hahn-echo coherence is exp(-1) at t = T2."""
        x = float(t2_echo_s) / float(correlation_time_s)
        shape = x - 3.0 + 4.0 * math.exp(-0.5 * x) - math.exp(-x)
        if shape <= 0:
            raise ValueError("Invalid Hahn-echo calibration parameters")
        sigma = 1.0 / (float(correlation_time_s) * math.sqrt(shape))
        return cls(
            correlation_time_s=float(correlation_time_s),
            sigma_rad_s=sigma,
            mean_rad_s=float(mean_rad_s),
            calibration="hahn_echo_t2",
            coherence_time_s=float(t2_echo_s),
        )

    def ramsey_coherence(self, time_s: np.ndarray | float) -> np.ndarray:
        """Analytic Ramsey/FID envelope for H_noise = beta(t) Z / 2."""
        t = np.asarray(time_s, dtype=float)
        tc = self.correlation_time_s
        chi = (self.sigma_rad_s * tc) ** 2 * (t / tc - 1.0 + np.exp(-t / tc))
        return np.exp(-chi)

    def hahn_echo_coherence(self, time_s: np.ndarray | float) -> np.ndarray:
        """Analytic Hahn-echo envelope for H_noise = beta(t) Z / 2."""
        t = np.asarray(time_s, dtype=float)
        tc = self.correlation_time_s
        x = t / tc
        chi = (self.sigma_rad_s * tc) ** 2 * (
            x - 3.0 + 4.0 * np.exp(-0.5 * x) - np.exp(-x)
        )
        return np.exp(-chi)


def sample_ou_process(
    time_grid_s: Sequence[float] | torch.Tensor | np.ndarray,
    params: OUParameters,
    *,
    n_realizations: int,
    seed: int | None = None,
    stationary_initial_state: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate exact discrete OU trajectories.

    Returns
    -------
    torch.Tensor
        Shape ``(n_realizations, len(time_grid_s))`` in rad/s.
    """
    if n_realizations <= 0:
        raise ValueError("n_realizations must be positive")

    times = torch.as_tensor(time_grid_s, dtype=torch.float64, device="cpu")
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("time_grid_s must be one-dimensional with at least two points")
    dts = times[1:] - times[:-1]
    if torch.any(dts <= 0):
        raise ValueError("time_grid_s must be strictly increasing")

    out_device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))

    trace = torch.empty((n_realizations, times.numel()), dtype=dtype, device="cpu")
    if stationary_initial_state:
        trace[:, 0] = params.mean_rad_s + params.sigma_rad_s * torch.randn(
            n_realizations, generator=generator, dtype=dtype
        )
    else:
        trace[:, 0] = params.mean_rad_s

    tc = params.correlation_time_s
    for k, dt in enumerate(dts.tolist()):
        a = math.exp(-dt / tc)
        innovation_std = params.sigma_rad_s * math.sqrt(max(0.0, 1.0 - a * a))
        xi = torch.randn(n_realizations, generator=generator, dtype=dtype)
        trace[:, k + 1] = (
            params.mean_rad_s
            + a * (trace[:, k] - params.mean_rad_s)
            + innovation_std * xi
        )

    return trace.to(device=out_device)


def sample_independent_channels(
    time_grid_s: Sequence[float] | torch.Tensor | np.ndarray,
    channels: Mapping[str, OUParameters],
    *,
    n_realizations: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Generate independent OU trajectories for named dephasing channels."""
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(channels))
    result: dict[str, torch.Tensor] = {}
    for (name, params), child in zip(channels.items(), child_sequences):
        channel_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        result[name] = sample_ou_process(
            time_grid_s,
            params,
            n_realizations=n_realizations,
            seed=channel_seed,
            device=device,
            dtype=dtype,
        )
    return result


def ensemble_gate_metrics(
    target: torch.Tensor,
    logical_kraus: Iterable[torch.Tensor],
) -> dict[str, np.ndarray | float]:
    """Compute ensemble-averaged logical gate metrics, including leakage.

    Each projected propagator K_r is a logical Kraus operator for one classical
    noise realization. The ensemble channel uses K_r / sqrt(N). The reported
    average fidelity is unconditioned: leakage lowers both survival and fidelity.
    """
    target = target.to(dtype=torch.complex128)
    d = int(target.shape[0])
    if target.shape != (d, d):
        raise ValueError("target must be square")

    f_ent_values: list[float] = []
    survival_values: list[float] = []
    f_avg_values: list[float] = []

    for K in logical_kraus:
        K = K.to(dtype=torch.complex128, device=target.device)
        if K.shape != target.shape:
            raise ValueError("All logical propagators must match target shape")
        overlap = torch.trace(target.conj().T @ K)
        f_ent = float((overlap.abs() ** 2 / (d * d)).real.item())
        survival = float((torch.trace(K.conj().T @ K).real / d).item())
        f_avg = (d * f_ent + survival) / (d + 1.0)
        f_ent_values.append(f_ent)
        survival_values.append(survival)
        f_avg_values.append(f_avg)

    if not f_ent_values:
        raise ValueError("logical_kraus is empty")

    f_ent_arr = np.asarray(f_ent_values)
    survival_arr = np.asarray(survival_values)
    f_avg_arr = np.asarray(f_avg_values)

    def stderr(x: np.ndarray) -> float:
        return float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0

    return {
        "entanglement_fidelity_samples": f_ent_arr,
        "average_gate_fidelity_samples": f_avg_arr,
        "survival_samples": survival_arr,
        "entanglement_fidelity_mean": float(f_ent_arr.mean()),
        "entanglement_fidelity_stderr": stderr(f_ent_arr),
        "average_gate_fidelity_mean": float(f_avg_arr.mean()),
        "average_gate_fidelity_stderr": stderr(f_avg_arr),
        "survival_mean": float(survival_arr.mean()),
        "survival_stderr": stderr(survival_arr),
        "average_gate_fidelity_quantiles": np.quantile(f_avg_arr, [0.05, 0.5, 0.95]),
    }
