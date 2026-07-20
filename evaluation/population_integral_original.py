"""Generate the original-style three-qubit population-integral figure.

The figure intentionally follows the plotting convention used in the original
repository: eight logical-basis population trajectories, one highlighted basis
state, a shaded integral, a two-column legend, and a compact annotation box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from noise.run import (
    checkpoint_path_from,
    load_checkpoint,
    resample_controls,
    validate_checkpoint,
)


LOGICAL_LABELS = [
    f"|{a}{b}{c}⟩"
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
]


def _state_index(bits: str) -> int:
    if len(bits) != 3 or any(bit not in "01" for bit in bits):
        raise ValueError("logical state must be a three-bit string such as 000 or 100")
    return int(bits, 2)


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def logical_population_trajectory(
    project_root: Path,
    checkpoint_value: Path,
    *,
    initial_state: str,
    steps_per_ns: float | None,
) -> dict[str, Any]:
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from quantum_model_NV import (  # type: ignore
        get_U_RWA,
        get_precomp,
        set_active_carbons,
        ω1,
    )

    checkpoint_path = checkpoint_path_from(resolve(project_root, checkpoint_value))
    checkpoint = load_checkpoint(checkpoint_path)
    time_grid, drive, basis_indices, delta_e = validate_checkpoint(
        checkpoint, checkpoint_path
    )
    time_grid, drive = resample_controls(
        time_grid, drive, steps_per_ns=steps_per_ns
    )

    set_active_carbons([1, 2])
    precomputed = get_precomp()
    dimension = 2 * 3 * (2 ** int(precomputed["N_C"]))
    initial_logical_index = _state_index(initial_state)
    initial_full_index = basis_indices[initial_logical_index]

    state = torch.zeros(dimension, dtype=torch.complex128)
    state[initial_full_index] = 1.0

    logical_index = torch.as_tensor(basis_indices, dtype=torch.long)
    populations: list[np.ndarray] = []
    survival: list[float] = []

    def sample(current: torch.Tensor) -> None:
        logical = current.index_select(0, logical_index)
        raw = torch.abs(logical).square().real
        weight = float(raw.sum().detach().cpu())
        survival.append(weight)
        if weight > 1.0e-15:
            raw = raw / weight
        populations.append(raw.detach().cpu().numpy())

    sample(state)
    for index in range(time_grid.numel() - 1):
        dt = float((time_grid[index + 1] - time_grid[index]).item())
        time = float(time_grid[index].item())
        controls = [channel[index] for channel in drive]
        step = get_U_RWA(controls, dt, time, Δ_e=delta_e, ω_RF=ω1)
        state = step @ state
        sample(state)

    return {
        "checkpoint": str(checkpoint_path),
        "time_ns": time_grid.detach().cpu().numpy() * 1.0e9,
        "populations": np.asarray(populations, dtype=float),
        "survival": np.asarray(survival, dtype=float),
        "initial_state": initial_state,
    }


def plot_original_style(
    data: dict[str, Any],
    *,
    gate: str,
    highlight_state: str,
    output: Path,
) -> dict[str, Any]:
    highlight_index = _state_index(highlight_state)
    time_ns = np.asarray(data["time_ns"], dtype=float)
    populations = np.asarray(data["populations"], dtype=float)
    highlighted = populations[:, highlight_index]
    integrate = getattr(np, "trapezoid", np.trapz)
    area_ns = float(integrate(highlighted, time_ns))

    fig = plt.figure(figsize=(10, 6))
    for index, label in enumerate(LOGICAL_LABELS):
        linewidth = 2.5 if index == highlight_index else 1.5
        plt.plot(time_ns, populations[:, index], label=label, linewidth=linewidth)

    plt.fill_between(
        time_ns,
        highlighted,
        alpha=0.25,
        zorder=0,
        label=f"Area under {LOGICAL_LABELS[highlight_index]}",
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title(f"{gate.upper()} population dynamics")
    plt.legend(ncol=2)
    plt.grid(True)

    text = (
        rf"$\int P_{{{LOGICAL_LABELS[highlight_index]}}}(t)\,dt$"
        + "\n"
        + rf"$= {area_ns:.3f}\,\mathrm{{ns}}$"
    )
    plt.text(
        0.98,
        0.2,
        text,
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="center",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    summary = {
        "gate": gate.lower(),
        "checkpoint": data["checkpoint"],
        "initial_state": data["initial_state"],
        "highlight_state": highlight_state,
        "highlight_integral_ns": area_ns,
        "minimum_logical_survival": float(np.min(data["survival"])),
        "final_logical_survival": float(data["survival"][-1]),
        "figure": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("zzz", "xzz"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory or pulse_solution.pt.",
    )
    parser.add_argument("--initial-state", default="000")
    parser.add_argument("--highlight-state", default="100")
    parser.add_argument("--steps-per-ns", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data = logical_population_trajectory(
        project_root,
        args.checkpoint,
        initial_state=args.initial_state,
        steps_per_ns=args.steps_per_ns,
    )
    summary = plot_original_style(
        data,
        gate=args.gate,
        highlight_state=args.highlight_state,
        output=resolve(project_root, args.output),
    )
    print(f"Figure: {summary['figure']}")
    print(
        f"Integral P_{args.highlight_state}: "
        f"{summary['highlight_integral_ns']:.6f} ns"
    )


if __name__ == "__main__":
    main()
