from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from .config import ObjectiveWeights


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


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    coordinates: dict[str, torch.Tensor]


def _kron3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.kron(torch.kron(a, b), c)


def _paulis(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    identity = torch.eye(2, dtype=dtype, device=device)
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    h = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device) / math.sqrt(2.0)
    return identity, x, z, h


class SupportSelectiveObjective:
    """Differentiable three-qubit support-selective terminal objective."""

    def __init__(
        self,
        *,
        basis_indices: list[int],
        gate: str,
        target_angle: float,
        pair_weight: float,
        tripartite_weight: float,
        weights: ObjectiveWeights,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.gate = gate.lower()
        if self.gate not in {"zzz", "xzz"}:
            raise ValueError("gate must be 'zzz' or 'xzz'.")
        self.target_angle = float(target_angle)
        self.pair_weight = float(pair_weight)
        self.tripartite_weight = float(tripartite_weight)
        self.weights = weights
        self.dtype = dtype
        self.real_dtype = torch.float64
        self.device = device
        self.basis_indices = torch.as_tensor(basis_indices, dtype=torch.long, device=device)

        identity, _, z, h = _paulis(dtype, device)
        self.z_a = _kron3(z, identity, identity)
        self.z_b = _kron3(identity, z, identity)
        self.z_c = _kron3(identity, identity, z)
        self.zzz = _kron3(z, z, z)
        self.frame = _kron3(h, identity, identity) if self.gate == "xzz" else torch.eye(
            8, dtype=dtype, device=device
        )
        self.target_frame = torch.linalg.matrix_exp(1j * self.target_angle * self.zzz)
        self.walsh = torch.tensor(_WALSH_ROWS, dtype=self.real_dtype, device=device) / 8.0

    def logical_block(self, full_propagator: torch.Tensor) -> torch.Tensor:
        return full_propagator.index_select(0, self.basis_indices).index_select(1, self.basis_indices)

    def framed_block(self, logical_block: torch.Tensor) -> torch.Tensor:
        return self.frame @ logical_block @ self.frame.conj().T

    def coordinates_from_framed(self, framed: torch.Tensor) -> dict[str, torch.Tensor]:
        # No explicit branch wrapping is needed: all terminal phase losses below
        # are pi-periodic through cos(2*delta), and nonempty Walsh rows remove
        # global phase automatically.
        basis_phases = torch.angle(torch.diagonal(framed))
        values = self.walsh @ basis_phases
        return dict(zip(_COORDINATE_NAMES, values))

    def local_corrected_fidelity(
        self, framed: torch.Tensor, coordinates: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        local_generator = (
            coordinates["A"] * self.z_a
            + coordinates["B"] * self.z_b
            + coordinates["C"] * self.z_c
        )
        local_correction = torch.linalg.matrix_exp(-1j * local_generator)
        corrected = local_correction @ framed
        overlap = torch.trace(self.target_frame.conj().T @ corrected)
        fidelity = overlap.abs().square().real / 64.0
        return torch.clamp(fidelity, min=0.0, max=1.0)

    def __call__(
        self,
        full_propagator: torch.Tensor,
        *,
        fluence: torch.Tensor,
        smoothness: torch.Tensor,
        peak_penalty: torch.Tensor | None = None,
        population_100_sum: torch.Tensor | None = None,
    ) -> ObjectiveResult:
        if population_100_sum is None:
            population_100_sum = torch.zeros(
                (), dtype=self.real_dtype, device=self.device
            )
        logical = self.logical_block(full_propagator)
        framed = self.framed_block(logical)
        coordinates = self.coordinates_from_framed(framed)

        off_diagonal = framed - torch.diag(torch.diagonal(framed))
        diagonality = off_diagonal.abs().square().sum().real / 8.0

        identity = torch.eye(8, dtype=logical.dtype, device=logical.device)
        gram = logical.conj().T @ logical
        unitarity = (gram - identity).abs().square().sum().real / 8.0
        survival = torch.trace(gram).real / 8.0
        survival_loss = torch.relu(1.0 - survival)

        selected_phase = torch.zeros((), dtype=self.real_dtype, device=self.device)
        selected_weight = 0.0
        for name in ("AB", "AC", "BC"):
            selected_phase = selected_phase + self.pair_weight * (
                1.0 - torch.cos(2.0 * coordinates[name])
            )
            selected_weight += self.pair_weight
        selected_phase = selected_phase + self.tripartite_weight * (
            1.0 - torch.cos(2.0 * (coordinates["ABC"] - self.target_angle))
        )
        selected_weight += self.tripartite_weight
        selected_phase = selected_phase / selected_weight

        corrected_fidelity = self.local_corrected_fidelity(framed, coordinates)
        corrected_infidelity = 1.0 - corrected_fidelity

        if peak_penalty is None:
            peak_penalty = torch.zeros((), dtype=self.real_dtype, device=self.device)

        components = {
            "corrected_infidelity": corrected_infidelity,
            "selected_phase": selected_phase,
            "diagonality": diagonality,
            "unitarity": unitarity,
            "survival_loss": survival_loss,
            "peak_penalty": peak_penalty,
            "fluence": fluence,
            "smoothness": smoothness,
            "population_100_sum": population_100_sum,
            "corrected_fidelity": corrected_fidelity,
            "survival": survival,
        }

        w = self.weights
        loss = (
            w.corrected_infidelity * corrected_infidelity
            + w.selected_phase * selected_phase
            + w.diagonality * diagonality
            + w.unitarity * unitarity
            + w.survival * survival_loss
            + w.peak * peak_penalty
            + w.fluence * fluence
            + w.smoothness * smoothness
            + w.population_100_sum * population_100_sum
        )
        return ObjectiveResult(loss=loss, components=components, coordinates=coordinates)
