from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class TrajectoryMetrics:
    propagator: torch.Tensor
    population_100_sum: torch.Tensor
    times: torch.Tensor | None = None
    population_100_trace: torch.Tensor | None = None
    logical_survival_trace: torch.Tensor | None = None


def _validate_inputs(
    time_grid: torch.Tensor,
    controls: list[torch.Tensor],
    basis_indices: list[int],
    dimension: int,
) -> None:
    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("time_grid must be one-dimensional with at least two points")
    if not controls:
        raise ValueError("at least one control channel is required")
    if any(control.shape != time_grid.shape for control in controls):
        raise ValueError("all controls must have the same shape as time_grid")
    if torch.any(torch.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if len(basis_indices) != 8 or len(set(basis_indices)) != 8:
        raise ValueError("basis_indices must contain eight unique logical indices")
    if min(basis_indices) < 0 or max(basis_indices) >= dimension:
        raise ValueError("logical basis index outside the full Hilbert space")


def propagate_with_population_100_sum(
    get_step: Callable[[list[torch.Tensor], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    controls: list[torch.Tensor],
    *,
    basis_indices: list[int],
    dimension: int,
    return_traces: bool = False,
) -> TrajectoryMetrics:
    """Propagate the gate and sum the plotted |100> population.

    The initial state is logical |000>. At every propagation sample, the
    logical populations are computed exactly as in
    ``evaluation.population_integral_original``: the eight logical
    populations are divided by their total logical survival whenever that
    survival is nonzero. The optimization penalty is then the discrete sum

        sum_k P_100(t_k).

    No electron-Z variance, opposite-manifold average, or averaging over other
    logical inputs is used.
    """
    _validate_inputs(time_grid, controls, basis_indices, dimension)

    device = controls[0].device
    dtype = torch.complex128
    propagator = torch.eye(dimension, dtype=dtype, device=device)
    logical_index = torch.as_tensor(basis_indices, dtype=torch.long, device=device)
    initial_full_index = int(basis_indices[0])
    target_logical_index = 4  # binary 100 in the standard logical ordering

    population_samples: list[torch.Tensor] = []
    survival_samples: list[torch.Tensor] = []
    sampled_times: list[torch.Tensor] = []

    def sample(current: torch.Tensor, time_value: torch.Tensor) -> None:
        state = current[:, initial_full_index]
        logical_state = state.index_select(0, logical_index)
        raw_populations = logical_state.abs().square().real
        survival = raw_populations.sum()
        normalized = torch.where(
            survival > 1.0e-15,
            raw_populations / torch.clamp(survival, min=1.0e-15),
            raw_populations,
        )
        population_samples.append(normalized[target_logical_index])
        survival_samples.append(survival)
        sampled_times.append(time_value)

    sample(propagator, time_grid[0])
    for index in range(time_grid.numel() - 1):
        time_value = time_grid[index]
        dt = float((time_grid[index + 1] - time_grid[index]).detach().cpu())
        channel_values = [control[index] for control in controls]
        propagator = get_step(
            channel_values,
            dt,
            float(time_value.detach().cpu()),
        ) @ propagator
        sample(propagator, time_grid[index + 1])

    population_trace = torch.stack(population_samples)
    survival_trace = torch.stack(survival_samples)
    return TrajectoryMetrics(
        propagator=propagator,
        population_100_sum=population_trace.sum(),
        times=torch.stack(sampled_times) if return_traces else None,
        population_100_trace=population_trace if return_traces else None,
        logical_survival_trace=survival_trace if return_traces else None,
    )
