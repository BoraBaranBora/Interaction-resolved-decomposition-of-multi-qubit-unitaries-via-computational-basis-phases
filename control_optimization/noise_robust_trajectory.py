from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

StepConvention = Literal["intervals", "samples"]


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
    step_convention: StepConvention,
) -> None:
    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("time_grid must be one-dimensional with at least two points")
    if not controls:
        raise ValueError("at least one control channel is required")
    if any(control.shape != time_grid.shape for control in controls):
        raise ValueError("all controls must have the same shape as time_grid")
    if electron_z.ndim != 2 or electron_z.shape[0] != electron_z.shape[1]:
        raise ValueError("electron_z must be a square matrix")
    if len(basis_indices) != 8 or len(set(basis_indices)) != 8:
        raise ValueError("basis_indices must contain eight unique logical indices")
    if min(basis_indices) < 0 or max(basis_indices) >= electron_z.shape[0]:
        raise ValueError("logical basis index outside the full Hilbert space")
    if step_convention not in {"intervals", "samples"}:
        raise ValueError("step_convention must be 'intervals' or 'samples'")


def _uniform_step(time_grid: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return the mean step for a numerically uniform grid.

    ``torch.linspace`` grids stored as float32 do not have bitwise-identical
    adjacent differences.  The accumulated rounding error scales with the
    number of points, so the tolerance is derived from machine precision and
    grid length.  This accepts harmless representation jitter while still
    rejecting a physically nonuniform propagation grid.
    """
    differences = torch.diff(time_grid)
    if torch.any(differences <= 0):
        raise ValueError("time_grid must be strictly increasing")

    dt_tensor = differences.mean()
    dt = float(dt_tensor.detach().cpu())
    if dt <= 0.0:
        raise ValueError("time_grid must be strictly increasing")

    if time_grid.dtype.is_floating_point:
        eps = float(torch.finfo(time_grid.dtype).eps)
    else:
        eps = 0.0

    max_relative_jitter = float(
        (torch.max(torch.abs(differences - dt_tensor)) / torch.abs(dt_tensor))
        .detach()
        .cpu()
    )
    rounding_tolerance = max(1.0e-10, 8.0 * eps * float(time_grid.numel()))

    if max_relative_jitter > rounding_tolerance:
        raise ValueError(
            "trajectory metric requires a numerically uniform time grid "
            f"(relative step variation {max_relative_jitter:.3e}, "
            f"allowed {rounding_tolerance:.3e})"
        )
    return dt_tensor, dt


def propagate_with_electron_metrics(
    get_step: Callable[[list[torch.Tensor], float, float], torch.Tensor],
    time_grid: torch.Tensor,
    controls: list[torch.Tensor],
    *,
    basis_indices: list[int],
    electron_z: torch.Tensor,
    sample_stride: int = 1,
    return_traces: bool = False,
    step_convention: StepConvention = "intervals",
) -> TrajectoryMetrics:
    """Propagate the unitary and evaluate electron-detuning exposure.

    For each logical computational-basis input ``x`` this routine evaluates

        Var_x[Z_A](t) = 1 - <Z_A>_x(t)^2

    and the population in the electron manifold opposite to the input state.
    The optimization scalar is the average of ``Var[Z_A]`` over time and over
    the eight logical basis inputs. Unlike the raw population of a fixed
    electron manifold, this quantity is not fixed by unitarity.

    ``step_convention='intervals'`` performs ``len(time_grid)-1`` steps,
    treating the grid as interval endpoints. ``step_convention='samples'``
    performs one step for every control sample. The runner detects which
    convention matches the repository's existing propagator once per run.
    """
    _validate_inputs(time_grid, controls, basis_indices, electron_z, step_convention)
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")

    device = electron_z.device
    dtype = electron_z.dtype
    dimension = electron_z.shape[0]
    dt_tensor, dt = _uniform_step(time_grid)

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
    n_steps = (
        time_grid.numel() - 1
        if step_convention == "intervals"
        else time_grid.numel()
    )
    for index in range(n_steps):
        t_value = time_grid[index]
        channel_values = [control[index] for control in controls]
        propagator = get_step(
            channel_values, dt, float(t_value.detach().cpu())
        ) @ propagator
        if (index + 1) % sample_stride == 0 or index + 1 == n_steps:
            sample_time = (
                time_grid[index + 1]
                if step_convention == "intervals"
                else t_value + dt_tensor
            )
            sample(propagator, sample_time)

    exposure_trace = torch.stack(exposure_samples)
    excursion_trace = torch.stack(excursion_samples)
    return TrajectoryMetrics(
        propagator=propagator,
        dephasing_exposure=exposure_trace.mean(),
        manifold_excursion=excursion_trace.mean(),
        times=torch.stack(sampled_times) if return_traces else None,
        exposure_trace=exposure_trace if return_traces else None,
        excursion_trace=excursion_trace if return_traces else None,
        excursion_min_trace=(
            torch.stack(excursion_min_samples) if return_traces else None
        ),
        excursion_max_trace=(
            torch.stack(excursion_max_samples) if return_traces else None
        ),
    )
