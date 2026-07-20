import torch

from control_optimization.trajectory import propagate_with_population_100_sum


def test_nonuniform_float_grid_uses_each_interval_exactly() -> None:
    grid = torch.tensor([0.0, 0.1, 0.20000002, 0.3], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]
    seen = []

    def get_step(_controls, dt, time):
        seen.append((dt, time))
        phase = torch.exp(torch.tensor(-1j * dt, dtype=torch.complex128))
        return phase * torch.eye(8, dtype=torch.complex128)

    result = propagate_with_population_100_sum(
        get_step,
        grid,
        controls,
        basis_indices=list(range(8)),
        dimension=8,
        return_traces=True,
    )
    expected_phase = torch.exp(
        torch.tensor(-1j * float(grid[-1]), dtype=torch.complex128)
    )
    assert torch.allclose(
        result.propagator,
        expected_phase * torch.eye(8, dtype=torch.complex128),
    )
    assert len(seen) == len(grid) - 1
    assert float(result.population_100_sum) == 0.0
