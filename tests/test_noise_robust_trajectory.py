import math
import torch
from control_optimization.noise_robust_trajectory import propagate_with_electron_metrics

def electron_z_8():
    return torch.diag(torch.tensor([1.0] * 4 + [-1.0] * 4, dtype=torch.complex128))

def test_identity_has_zero_exposure_and_excursion():
    grid = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]
    def get_step(_controls, _dt, _time):
        return torch.eye(8, dtype=torch.complex128)
    result = propagate_with_electron_metrics(get_step, grid, controls, basis_indices=list(range(8)), electron_z=electron_z_8(), return_traces=True)
    assert float(result.dephasing_exposure) == 0.0
    assert float(result.manifold_excursion) == 0.0

def test_half_rotation_creates_exposure():
    grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    controls = [torch.zeros_like(grid)]
    x = torch.zeros((8, 8), dtype=torch.complex128)
    for i in range(4):
        x[i, i+4] = 1.0
        x[i+4, i] = 1.0
    step = torch.linalg.matrix_exp(-1j * (math.pi / 4.0) * x)
    def get_step(_controls, _dt, _time):
        return step
    result = propagate_with_electron_metrics(get_step, grid, controls, basis_indices=list(range(8)), electron_z=electron_z_8(), return_traces=True)
    assert float(result.dephasing_exposure) > 0.4
    assert float(result.manifold_excursion) > 0.2
