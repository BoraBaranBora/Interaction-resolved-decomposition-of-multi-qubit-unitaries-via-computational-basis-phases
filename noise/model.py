"""Logical-frame utilities for the electronic dephasing calculation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import math
import numpy as np
import torch


_COORDINATE_NAMES = ("A", "B", "C", "AB", "AC", "BC", "ABC")
_WALSH_ROWS = (
    (1, 1, 1, 1, -1, -1, -1, -1),
    (1, 1, -1, -1, 1, 1, -1, -1),
    (1, -1, 1, -1, 1, -1, 1, -1),
    (1, 1, -1, -1, -1, -1, 1, 1),
    (1, -1, 1, -1, -1, 1, -1, 1),
    (1, -1, -1, 1, 1, -1, -1, 1),
    (1, -1, -1, 1, -1, 1, 1, -1),
)


def kron3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.kron(torch.kron(a, b), c)


def paulis(
    *, dtype: torch.dtype = torch.complex128, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(2, dtype=dtype, device=device)
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    h = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device) / math.sqrt(2.0)
    return identity, x, z, h


def frame_for_gate(
    gate: str,
    *,
    dtype: torch.dtype = torch.complex128,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    identity, _, _, hadamard = paulis(dtype=dtype, device=device)
    normalized = gate.lower()
    if normalized in {"diagonal", "zzz"}:
        return torch.eye(8, dtype=dtype, device=device)
    if normalized in {"nondiagonal", "xzz"}:
        return kron3(hadamard, identity, identity)
    raise ValueError("gate must be diagonal/ZZZ or nondiagonal/XZZ")


def target_in_diagonal_frame(
    *,
    angle_rad: float = math.pi / 4.0,
    dtype: torch.dtype = torch.complex128,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    _, _, z, _ = paulis(dtype=dtype, device=device)
    return torch.linalg.matrix_exp(1j * float(angle_rad) * kron3(z, z, z))


def support_phase_coordinates(framed_logical: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the seven non-global Walsh phase coordinates."""
    if framed_logical.shape != (8, 8):
        raise ValueError("framed_logical must be an 8x8 matrix")
    phases = torch.angle(torch.diagonal(framed_logical))
    walsh = torch.tensor(
        _WALSH_ROWS, dtype=phases.dtype, device=phases.device
    ) / 8.0
    values = walsh @ phases
    return dict(zip(_COORDINATE_NAMES, values))


def fixed_local_correction(framed_nominal: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calibrate one fixed local correction from the noiseless pulse.

    The convention exactly matches ``SupportSelectiveObjective``: the framed
    propagator is left-multiplied by exp[-i(phi_A Z_A + phi_B Z_B + phi_C Z_C)].
    """
    coordinates = support_phase_coordinates(framed_nominal)
    identity, _, z, _ = paulis(
        dtype=framed_nominal.dtype, device=framed_nominal.device
    )
    generator = (
        coordinates["A"] * kron3(z, identity, identity)
        + coordinates["B"] * kron3(identity, z, identity)
        + coordinates["C"] * kron3(identity, identity, z)
    )
    correction = torch.linalg.matrix_exp(-1j * generator)
    return correction, coordinates


def electronic_dephasing_operator(
    qz: torch.Tensor,
    dim_nuc: int,
    *,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Embed Z_A/2 in the complete electron-nuclear Hilbert space."""
    if int(dim_nuc) <= 0:
        raise ValueError("dim_nuc must be positive")
    qz = torch.as_tensor(qz, dtype=dtype)
    if qz.shape != (2, 2):
        raise ValueError("qz must be the 2x2 electron Pauli-Z operator")
    operator = torch.kron(0.5 * qz, torch.eye(int(dim_nuc), dtype=dtype))
    if not torch.allclose(operator, operator.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("embedded electronic dephasing operator is not Hermitian")
    return operator


def logical_block(full_propagator: torch.Tensor, basis_indices: list[int]) -> torch.Tensor:
    if len(basis_indices) != 8 or len(set(basis_indices)) != 8:
        raise ValueError("checkpoint basis_indices must contain 8 unique entries")
    index = torch.as_tensor(
        basis_indices, dtype=torch.long, device=full_propagator.device
    )
    return full_propagator.index_select(0, index).index_select(1, index)


def corrected_logical_operator(
    full_propagator: torch.Tensor,
    *,
    basis_indices: list[int],
    frame: torch.Tensor,
    correction: torch.Tensor,
) -> torch.Tensor:
    logical = logical_block(full_propagator, basis_indices)
    framed = frame @ logical @ frame.conj().T
    return correction @ framed


def logical_gate_metrics(
    target: torch.Tensor,
    corrected_operators: Iterable[torch.Tensor],
) -> dict[str, Any]:
    """Compute leakage-aware realization and ensemble gate metrics."""
    target = torch.as_tensor(target, dtype=torch.complex128)
    if target.shape != (8, 8):
        raise ValueError("target must be 8x8")
    d = 8

    entanglement: list[float] = []
    survival: list[float] = []
    average: list[float] = []
    for operator in corrected_operators:
        operator = torch.as_tensor(
            operator, dtype=torch.complex128, device=target.device
        )
        if operator.shape != (8, 8):
            raise ValueError("all corrected operators must be 8x8")
        overlap = torch.trace(target.conj().T @ operator)
        f_ent = float((overlap.abs().square() / (d * d)).real.item())
        p_survival = float(
            (torch.trace(operator.conj().T @ operator).real / d).item()
        )
        f_avg = (d * f_ent + p_survival) / (d + 1.0)
        entanglement.append(f_ent)
        survival.append(p_survival)
        average.append(f_avg)

    if not average:
        raise ValueError("corrected_operators may not be empty")

    f_ent_arr = np.asarray(entanglement, dtype=float)
    survival_arr = np.asarray(survival, dtype=float)
    f_avg_arr = np.asarray(average, dtype=float)

    def standard_error(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        return float(values.std(ddof=1) / math.sqrt(values.size))

    return {
        "entanglement_fidelity_samples": f_ent_arr,
        "average_gate_fidelity_samples": f_avg_arr,
        "survival_samples": survival_arr,
        "entanglement_fidelity_mean": float(f_ent_arr.mean()),
        "entanglement_fidelity_stderr": standard_error(f_ent_arr),
        "average_gate_fidelity_mean": float(f_avg_arr.mean()),
        "average_gate_fidelity_stderr": standard_error(f_avg_arr),
        "survival_mean": float(survival_arr.mean()),
        "survival_stderr": standard_error(survival_arr),
        "average_gate_fidelity_quantiles_05_50_95": np.quantile(
            f_avg_arr, [0.05, 0.50, 0.95]
        ).tolist(),
    }
