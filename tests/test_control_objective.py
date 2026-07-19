import math

import torch

from control_optimization.config import ObjectiveWeights
from control_optimization.objective import SupportSelectiveObjective


def make_objective(gate: str = "zzz") -> SupportSelectiveObjective:
    return SupportSelectiveObjective(
        basis_indices=list(range(8)),
        gate=gate,
        target_angle=math.pi / 4,
        pair_weight=0.2,
        tripartite_weight=0.5,
        weights=ObjectiveWeights(
            corrected_infidelity=1.0,
            selected_phase=1.0,
            diagonality=1.0,
            unitarity=1.0,
            survival=1.0,
            fluence=0.0,
            smoothness=0.0,
        ),
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )


def zero() -> torch.Tensor:
    return torch.zeros((), dtype=torch.float64)


def test_target_has_zero_terminal_loss() -> None:
    objective = make_objective("zzz")
    result = objective(objective.target_frame, fluence=zero(), smoothness=zero())
    assert result.loss.item() < 1.0e-12
    assert abs(result.coordinates["ABC"].item() - math.pi / 4) < 1.0e-12


def test_local_phases_are_analytically_corrected() -> None:
    objective = make_objective("zzz")
    local = torch.linalg.matrix_exp(
        1j * (0.31 * objective.z_a - 0.22 * objective.z_b + 0.17 * objective.z_c)
    )
    unitary = local @ objective.target_frame
    result = objective(unitary, fluence=zero(), smoothness=zero())
    assert result.components["selected_phase"].item() < 1.0e-12
    assert result.components["corrected_infidelity"].abs().item() < 1.0e-12


def test_pairwise_phase_is_penalized() -> None:
    objective = make_objective("zzz")
    z_ab = objective.z_a @ objective.z_b
    unitary = torch.linalg.matrix_exp(1j * 0.15 * z_ab) @ objective.target_frame
    result = objective(unitary, fluence=zero(), smoothness=zero())
    assert result.coordinates["AB"].abs().item() > 0.1
    assert result.components["selected_phase"].item() > 0.0
    assert result.components["corrected_infidelity"].item() > 0.0


def test_xzz_is_diagonalized_by_hadamard_frame() -> None:
    objective = make_objective("xzz")
    physical_target = objective.frame.conj().T @ objective.target_frame @ objective.frame
    result = objective(physical_target, fluence=zero(), smoothness=zero())
    assert result.loss.item() < 1.0e-12


def test_phase_objective_has_autograd_gradient() -> None:
    objective = make_objective("zzz")
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    unitary = torch.linalg.matrix_exp(1j * theta.to(torch.complex128) * objective.zzz)
    result = objective(unitary, fluence=zero(), smoothness=zero())
    result.loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad)
    assert abs(theta.grad.item()) > 1.0e-6
