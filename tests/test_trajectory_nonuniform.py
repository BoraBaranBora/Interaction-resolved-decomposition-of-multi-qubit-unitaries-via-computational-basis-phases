import torch

from control_optimization.trajectory import propagate_with_electron_metrics


def test_nonuniform_float_grid_uses_each_interval_exactly():
    grid = torch.tensor([0.0, 0.1, 0.20000002, 0.3], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]
    z = torch.diag(torch.tensor([1.0] * 4 + [-1.0] * 4, dtype=torch.complex128))
    seen = []

    def get_step(_controls, dt, t):
        seen.append((dt, t))
        phase = torch.exp(torch.tensor(-1j * dt, dtype=torch.complex128))
        return phase * torch.eye(8, dtype=torch.complex128)

    result = propagate_with_electron_metrics(
        get_step,
        grid,
        controls,
        basis_indices=list(range(8)),
        electron_z=z,
        return_traces=True,
    )
    expected_phase = torch.exp(torch.tensor(-1j * float(grid[-1]), dtype=torch.complex128))
    assert torch.allclose(result.propagator, expected_phase * torch.eye(8, dtype=torch.complex128))
    assert len(seen) == len(grid) - 1
    assert float(result.dephasing_exposure) == 0.0
