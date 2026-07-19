"""Stationary Ornstein--Uhlenbeck electronic detuning noise.

The stochastic detuning beta(t) is expressed in angular-frequency units
(rad/s) and obeys

    d beta = -beta dt / tau_c + sqrt(2 sigma^2 / tau_c) dW.

For H_noise(t) = beta(t) Z / 2, the Ramsey coherence is

    W_R(t) = exp[-sigma^2 tau_c^2 (t/tau_c - 1 + exp(-t/tau_c))].

The production workflow accepts source-native experimental calibrations and
converts them once into the common pair (sigma, tau_c). No echo or dynamical-
decoupling pulse is inserted into the gate propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import numpy as np
import torch


def _ramsey_shape(x: float) -> float:
    return x + math.expm1(-x)


def _echo_shape(x: float) -> float:
    if abs(x) < 1.0e-3:
        return (
            x**3 / 12.0
            - x**4 / 32.0
            + 7.0 * x**5 / 960.0
            - x**6 / 768.0
            + 31.0 * x**7 / 161280.0
        )
    return x - 3.0 + 4.0 * math.exp(-x / 2.0) - math.exp(-x)


@dataclass(frozen=True)
class OUParameters:
    """Parameters of a stationary, zero-mean OU frequency process."""

    correlation_time_s: float
    sigma_rad_s: float
    equivalent_t2_star_s: float
    calibration: str

    def __post_init__(self) -> None:
        for name, value in (
            ("correlation_time_s", self.correlation_time_s),
            ("sigma_rad_s", self.sigma_rad_s),
            ("equivalent_t2_star_s", self.equivalent_t2_star_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.calibration:
            raise ValueError("calibration must be nonempty")

    @classmethod
    def from_t2_star(
        cls,
        t2_star_s: float,
        correlation_time_s: float,
        *,
        calibration: str = "ramsey_t2_star",
    ) -> "OUParameters":
        """Set sigma by requiring W_R(T2*) = exp(-1)."""
        t2 = float(t2_star_s)
        tau = float(correlation_time_s)
        if not math.isfinite(t2) or t2 <= 0.0:
            raise ValueError("t2_star_s must be finite and positive")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("correlation_time_s must be finite and positive")
        shape = _ramsey_shape(t2 / tau)
        if shape <= 0.0:
            raise ValueError("Ramsey calibration produced a non-positive shape")
        sigma = 1.0 / (tau * math.sqrt(shape))
        return cls(tau, sigma, t2, calibration)

    @classmethod
    def from_sigma(
        cls,
        sigma_rad_s: float,
        correlation_time_s: float,
        *,
        calibration: str = "source_native_sigma",
    ) -> "OUParameters":
        """Construct from an rms angular-frequency width and compute T2*."""
        sigma = float(sigma_rad_s)
        tau = float(correlation_time_s)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_rad_s must be finite and positive")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("correlation_time_s must be finite and positive")

        def chi(time_s: float) -> float:
            return (sigma * tau) ** 2 * _ramsey_shape(time_s / tau)

        lower, upper = 0.0, max(1.0 / sigma, 1.0e-12)
        while chi(upper) < 1.0:
            upper *= 2.0
            if upper > 1.0e6:
                raise RuntimeError("Could not bracket equivalent Ramsey time")
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            if chi(midpoint) < 1.0:
                lower = midpoint
            else:
                upper = midpoint
        t2 = 0.5 * (lower + upper)
        return cls(tau, sigma, t2, calibration)

    @classmethod
    def from_ramsey_and_echo(
        cls,
        t2_star_s: float,
        echo_t2_s: float,
        *,
        calibration: str = "ramsey_echo_ou_inference",
    ) -> "OUParameters":
        """Infer sigma and tau_c from W_R(T2*)=W_E(T2)=exp(-1).

        Echo data are used only to infer the environmental parameters. No echo
        pulse is inserted into the gate simulation.
        """
        t_star = float(t2_star_s)
        t_echo = float(echo_t2_s)
        if not (math.isfinite(t_star) and t_star > 0.0):
            raise ValueError("t2_star_s must be finite and positive")
        if not (math.isfinite(t_echo) and t_echo > t_star):
            raise ValueError("echo_t2_s must be finite and greater than T2*")

        # Eliminate sigma using the Ramsey equation and solve the ratio of
        # echo and Ramsey shape factors for tau on a logarithmic bracket.
        def residual(log_tau: float) -> float:
            tau = math.exp(log_tau)
            rshape = _ramsey_shape(t_star / tau)
            eshape = _echo_shape(t_echo / tau)
            return math.log(eshape / rshape)

        lower = math.log(t_star * 1.0e-4)
        upper = math.log(t_echo * 1.0e8)
        f_lower = residual(lower)
        f_upper = residual(upper)
        if f_lower == 0.0:
            log_tau = lower
        elif f_upper == 0.0:
            log_tau = upper
        elif f_lower * f_upper > 0.0:
            raise RuntimeError("Could not bracket OU correlation time")
        else:
            for _ in range(160):
                midpoint = 0.5 * (lower + upper)
                f_mid = residual(midpoint)
                if f_lower * f_mid <= 0.0:
                    upper = midpoint
                    f_upper = f_mid
                else:
                    lower = midpoint
                    f_lower = f_mid
            log_tau = 0.5 * (lower + upper)

        tau = math.exp(log_tau)
        rshape = _ramsey_shape(t_star / tau)
        sigma = 1.0 / (tau * math.sqrt(rshape))
        return cls(tau, sigma, t_star, calibration)

    def ramsey_coherence(self, time_s: float | np.ndarray) -> np.ndarray:
        t = np.asarray(time_s, dtype=float)
        tau = self.correlation_time_s
        x = t / tau
        chi = (self.sigma_rad_s * tau) ** 2 * (x - 1.0 + np.exp(-x))
        return np.exp(-chi)

    def echo_coherence(self, time_s: float | np.ndarray) -> np.ndarray:
        t = np.asarray(time_s, dtype=float)
        tau = self.correlation_time_s
        x = t / tau
        shape = x - 3.0 + 4.0 * np.exp(-x / 2.0) - np.exp(-x)
        small = np.abs(x) < 1.0e-3
        if np.any(small):
            xs = x[small] if x.ndim else x
            series = (
                xs**3 / 12.0 - xs**4 / 32.0 + 7.0 * xs**5 / 960.0
                - xs**6 / 768.0 + 31.0 * xs**7 / 161280.0
            )
            if x.ndim:
                shape = np.asarray(shape)
                shape[small] = series
            else:
                shape = series
        chi = (self.sigma_rad_s * tau) ** 2 * shape
        return np.exp(-chi)


def sample_ou_process(
    time_grid_s: Sequence[float] | np.ndarray | torch.Tensor,
    params: OUParameters,
    *,
    n_realizations: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate exact discrete OU trajectories on a nonuniform grid."""
    if int(n_realizations) <= 0:
        raise ValueError("n_realizations must be positive")

    times = torch.as_tensor(time_grid_s, dtype=torch.float64, device="cpu")
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("time_grid_s must be one-dimensional with >= 2 points")
    dts = times[1:] - times[:-1]
    if torch.any(dts <= 0.0):
        raise ValueError("time_grid_s must be strictly increasing")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = torch.empty((int(n_realizations), times.numel()), dtype=dtype)
    result[:, 0] = params.sigma_rad_s * torch.randn(
        int(n_realizations), dtype=dtype, generator=generator
    )

    tau = params.correlation_time_s
    for index, dt in enumerate(dts.tolist()):
        decay = math.exp(-dt / tau)
        innovation = params.sigma_rad_s * math.sqrt(max(0.0, 1.0 - decay * decay))
        xi = torch.randn(int(n_realizations), dtype=dtype, generator=generator)
        result[:, index + 1] = decay * result[:, index] + innovation * xi

    return result.to(device=device)
