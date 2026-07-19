from __future__ import annotations

import math

import torch

from noise.model import (
    electronic_dephasing_operator,
    fixed_local_correction,
    logical_gate_metrics,
    paulis,
    target_in_diagonal_frame,
)
from noise.run import propagate


def test_electronic_operator_is_embedded_z_over_two() -> None:
    qz = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    operator = electronic_dephasing_operator(qz, 6)
    assert operator.shape == (12, 12)
    assert torch.allclose(operator, operator.conj().T)
    eigenvalues = torch.linalg.eigvalsh(operator).real
    assert torch.allclose(eigenvalues[:6], -0.5 * torch.ones(6, dtype=eigenvalues.dtype))
    assert torch.allclose(eigenvalues[6:], 0.5 * torch.ones(6, dtype=eigenvalues.dtype))


def test_fixed_local_correction_matches_optimizer_convention() -> None:
    identity, _, z, _ = paulis()
    z_a = torch.kron(torch.kron(z, identity), identity)
    z_b = torch.kron(torch.kron(identity, z), identity)
    z_c = torch.kron(torch.kron(identity, identity), z)
    target = target_in_diagonal_frame()
    local = torch.linalg.matrix_exp(
        1j * (0.31 * z_a - 0.22 * z_b + 0.08 * z_c)
    )
    nominal = local @ target
    correction, coordinates = fixed_local_correction(nominal)
    corrected = correction @ nominal
    metrics = logical_gate_metrics(target, [corrected])
    assert math.isclose(metrics["entanglement_fidelity_mean"], 1.0, abs_tol=1e-12)
    assert math.isclose(float(coordinates["A"]), 0.31, abs_tol=1e-12)
    assert math.isclose(float(coordinates["B"]), -0.22, abs_tol=1e-12)
    assert math.isclose(float(coordinates["C"]), 0.08, abs_tol=1e-12)


def test_propagation_inserts_beta_z_over_two_at_every_step() -> None:
    seen: list[torch.Tensor] = []

    def fake_get_u(controls, dt, time, **kwargs):
        del controls, dt, time
        seen.append(kwargs["H_noise"].detach().clone())
        return torch.eye(4, dtype=torch.complex128)

    time = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    drive = [torch.zeros(3, dtype=torch.float64)]
    z_half = torch.diag(torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.complex128))
    beta = torch.tensor([2.0, -3.0, 9.0], dtype=torch.float64)
    result = propagate(
        get_u=fake_get_u,
        time_grid=time,
        drive=drive,
        delta_e=0.0,
        omega_rf=0.0,
        electron_z_half=z_half,
        beta_trace=beta,
    )
    assert torch.allclose(result, torch.eye(4, dtype=torch.complex128))
    assert len(seen) == 2
    assert torch.allclose(seen[0], 2.0 * z_half)
    assert torch.allclose(seen[1], -3.0 * z_half)


def test_control_resampling_uses_requested_uniform_grid() -> None:
    from noise.run import resample_controls

    time = torch.tensor([0.0, 0.5e-9, 1.5e-9, 2.0e-9], dtype=torch.float64)
    drive = [torch.tensor([0.0, 0.5, 1.5, 2.0], dtype=torch.float64)]
    new_time, new_drive = resample_controls(time, drive, steps_per_ns=1.0)
    assert torch.allclose(new_time, torch.tensor([0.0, 1e-9, 2e-9], dtype=torch.float64))
    assert torch.allclose(new_drive[0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
