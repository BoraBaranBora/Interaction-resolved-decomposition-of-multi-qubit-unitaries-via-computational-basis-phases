from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class TrajectoryMetrics:
    propagator: torch.Tensor
    dephasing_exposure: torch.Tensor
    manifold_excursion: torch.Tensor
    times: torch.Tensor | None = None
    exposure_trace: torch.Tensor | None = None
    excursion_trace: torch.Tensor | None = None
    excursion_min_trace: torch.Tensor | None = None
    excursion_max_trace: torch.Tensor | None = None


def _validate_inputs(
    time_grid: torch.Tensor,
    controls: list[torch.Tensor],
    basis_indices: list[int],
    electron_z: torch.Tensor,
) -> None:
    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("time_grid must be a one-dimensional tensor with at least two points")
    if not controls:
        raise ValueError("at least one control channel is required")
    if any(control.shape != time_grid.shape for control in controls):
        raise ValueError("all controls must have the same shape as time_grid")
    if torch.any(torch.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")
    if electron_z.ndim != 2 or electron_z.shape[0] != electron_z.shape[1]:
        raise ValueError("electron_z must be a square matrix")
    if len(basis_indices) != 8 or len(set(basis_indices)) != 8:
        raise ValueError("basis_indices must contain eight unique logical indices")
    if min(basis_indices) < 0 or max(basis_indices) >= electron_z.shape[0]:
        raise ValueError("logical basis index outside the full Hilbert space")


def _normalized_trapezoid(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    duration = times[-1] - times[0]
    if duration <= 0:
        raise ValueError("sampled trajectory duration must be positive")
    return torch.trapezoid(values, times) / duration


def propagate_with_electron_metrics(
    get_step: Callable[[list[torch.Tensor], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    controls: list[torch.Tensor],
    *,
    basis_indices: list[int],
    electron_z: torch.Tensor,
    sample_stride: int = 1,
    return_traces: bool = False,
) -> TrajectoryMetrics:
    """Mirror ``evolution.get_propagator`` and accumulate electron exposure.

    The repository propagator uses every interval ``[t_k,t_{k+1}]`` with its
    own ``dt_k`` and the control sampled at ``t_k``.  This routine follows that
    convention exactly, including the small floating-point variations produced
    by ``torch.linspace``.  No uniform-grid assumption or convention detection
    is used.
    """
    _validate_inputs(time_grid, controls, basis_indices, electron_z)
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")

    device = electron_z.device
    dtype = electron_z.dtype
    dimension = electron_z.shape[0]
    propagator = torch.eye(dimension, dtype=dtype, device=device)

    logical_index = torch.as_tensor(basis_indices, dtype=torch.long, device=device)
    initial_sign = torch.diagonal(electron_z).index_select(0, logical_index).real
    if not torch.allclose(
        initial_sign.abs(), torch.ones_like(initial_sign), atol=1.0e-10, rtol=0.0
    ):
        raise ValueError("logical basis states must be electron-Z eigenstates")

    exposure_samples: list[torch.Tensor] = []
    excursion_samples: list[torch.Tensor] = []
    excursion_min_samples: list[torch.Tensor] = []
    excursion_max_samples: list[torch.Tensor] = []
    sampled_times: list[torch.Tensor] = []

    def sample(current: torch.Tensor, t_value: torch.Tensor) -> None:
        states = current.index_select(1, logical_index)
        expectation = torch.sum(states.conj() * (electron_z @ states), dim=0).real
        expectation = torch.clamp(expectation, min=-1.0, max=1.0)
        variance = torch.clamp(1.0 - expectation.square(), min=0.0)
        opposite = torch.clamp(
            0.5 * (1.0 - initial_sign * expectation), min=0.0, max=1.0
        )
        exposure_samples.append(variance.mean())
        excursion_samples.append(opposite.mean())
        excursion_min_samples.append(opposite.min())
        excursion_max_samples.append(opposite.max())
        sampled_times.append(t_value)

    sample(propagator, time_grid[0])
    for index in range(time_grid.numel() - 1):
        t_value = time_grid[index]
        dt_tensor = time_grid[index + 1] - time_grid[index]
        dt = float(dt_tensor.detach().cpu())
        channel_values = [control[index] for control in controls]
        propagator = get_step(
            channel_values, dt, float(t_value.detach().cpu())
        ) @ propagator
        if (index + 1) % sample_stride == 0 or index + 1 == time_grid.numel() - 1:
            sample(propagator, time_grid[index + 1])

    sampled_time_tensor = torch.stack(sampled_times)
    exposure_trace = torch.stack(exposure_samples)
    excursion_trace = torch.stack(excursion_samples)

    return TrajectoryMetrics(
        propagator=propagator,
        dephasing_exposure=_normalized_trapezoid(exposure_trace, sampled_time_tensor),
        manifold_excursion=_normalized_trapezoid(excursion_trace, sampled_time_tensor),
        times=sampled_time_tensor if return_traces else None,
        exposure_trace=exposure_trace if return_traces else None,
        excursion_trace=excursion_trace if return_traces else None,
        excursion_min_trace=torch.stack(excursion_min_samples) if return_traces else None,
        excursion_max_trace=torch.stack(excursion_max_samples) if return_traces else None,
    )
