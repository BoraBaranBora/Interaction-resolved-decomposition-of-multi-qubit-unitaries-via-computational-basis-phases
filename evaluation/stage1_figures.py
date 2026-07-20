"""Generate stage-one pulse and interaction-phase trajectory figures."""

from __future__ import annotations

import argparse
import csv
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

from noise.run import checkpoint_path_from, load_checkpoint, resample_controls, validate_checkpoint

WALSH = np.asarray(
    [
        (1, 1, 1, 1, -1, -1, -1, -1),
        (1, 1, -1, -1, 1, 1, -1, -1),
        (1, -1, 1, -1, 1, -1, 1, -1),
        (1, 1, -1, -1, -1, -1, 1, 1),
        (1, -1, 1, -1, -1, 1, -1, 1),
        (1, -1, -1, 1, 1, -1, -1, 1),
        (1, -1, -1, 1, -1, 1, 1, -1),
    ],
    dtype=float,
) / 8.0
NAMES = ("A", "B", "C", "AB", "AC", "BC", "ABC")
LABELS = {
    "A": r"$\phi(A)$",
    "B": r"$\phi(B)$",
    "C": r"$\phi(C)$",
    "AB": r"$\phi(AB)$",
    "AC": r"$\phi(AC)$",
    "BC": r"$\phi(BC)$",
    "ABC": r"$\phi(ABC)$",
}


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def phase_trajectory(
    project_root: Path,
    checkpoint_value: Path,
    *,
    gate: str,
    steps_per_ns: float | None,
) -> dict[str, Any]:
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from quantum_model_NV import get_U_RWA, get_precomp, set_active_carbons, ω1

    checkpoint_path = checkpoint_path_from(resolve(project_root, checkpoint_value))
    checkpoint = load_checkpoint(checkpoint_path)
    time_grid, drive, basis_indices, delta_e = validate_checkpoint(checkpoint, checkpoint_path)
    time_grid, drive = resample_controls(time_grid, drive, steps_per_ns=steps_per_ns)

    set_active_carbons([1, 2])
    precomputed = get_precomp()
    dimension = 2 * 3 * (2 ** int(precomputed["N_C"]))
    propagator = torch.eye(dimension, dtype=torch.complex128)
    logical_index = torch.as_tensor(basis_indices, dtype=torch.long)

    identity = torch.eye(2, dtype=torch.complex128)
    hadamard = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2.0)
    frame = (
        torch.kron(torch.kron(hadamard, identity), identity)
        if gate.lower() == "xzz"
        else torch.eye(8, dtype=torch.complex128)
    )

    diagonal_phases: list[np.ndarray] = []

    def sample(current: torch.Tensor) -> None:
        logical = current.index_select(0, logical_index).index_select(1, logical_index)
        framed = frame @ logical @ frame.conj().T
        diagonal_phases.append(torch.angle(torch.diagonal(framed)).detach().cpu().numpy())

    sample(propagator)
    for index in range(time_grid.numel() - 1):
        dt = float((time_grid[index + 1] - time_grid[index]).item())
        time = float(time_grid[index].item())
        controls = [channel[index] for channel in drive]
        propagator = get_U_RWA(controls, dt, time, Δ_e=delta_e, ω_RF=ω1) @ propagator
        sample(propagator)

    unwrapped = np.unwrap(np.asarray(diagonal_phases, dtype=float), axis=0)
    coordinates = unwrapped @ WALSH.T
    return {
        "checkpoint": str(checkpoint_path),
        "time_ns": time_grid.detach().cpu().numpy() * 1.0e9,
        "drive_uT": drive[0].detach().cpu().numpy(),
        "coordinates": coordinates,
        "gate": gate.lower(),
    }


def save_figures(data: dict[str, Any], output_dir: Path) -> dict[str, str]:
    gate = str(data["gate"]).upper()
    time_ns = np.asarray(data["time_ns"], dtype=float)
    drive_uT = np.asarray(data["drive_uT"], dtype=float)
    coordinates = np.asarray(data["coordinates"], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The checkpoint stores the microwave-field amplitude in microtesla.
    # Using gamma_e/(2*pi) = 28.024 GHz/T,
    #
    #     Omega(t)/(2*pi) [MHz]
    #         = (gamma_e/(2*pi)) B(t)
    #         = 0.028024 B(t) [microtesla].
    gamma_e_over_2pi_mhz_per_uT = 0.028024
    omega_over_2pi_mhz = gamma_e_over_2pi_mhz_per_uT * drive_uT

    pulse_path = output_dir / f"stage1_{gate}_pulse.png"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time_ns, omega_over_2pi_mhz, linewidth=1.7)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"$\Omega(t)$ [$2\pi\times\mathrm{MHz}$]")
    ax.set_title(f"{gate} optimized pulse")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(pulse_path, dpi=200)
    plt.close(fig)

    phase_path = output_dir / f"stage1_{gate}_phase_trajectories.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, name in enumerate(NAMES):
        kwargs = {"linewidth": 2.6, "color": "black"} if name == "ABC" else {"linewidth": 1.4}
        ax.plot(time_ns, coordinates[:, index], label=LABELS[name], **kwargs)
    target_phase = -math.pi / 4.0 if gate == "ZZZ" else math.pi / 4.0
    target_label = r"$-\pi/4$" if gate == "ZZZ" else r"$\pi/4$"
    ax.axhline(
        target_phase,
        linestyle="--",
        color="black",
        alpha=0.6,
        label=target_label,
    )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Phase coefficient (rad)")
    ax.set_title(f"{gate} interaction-phase trajectories")
    ax.grid(True)
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(phase_path, dpi=200)
    plt.close(fig)

    csv_path = output_dir / f"stage1_{gate}_phase_trajectories.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_ns", *NAMES])
        for time, row in zip(time_ns, coordinates):
            writer.writerow([float(time), *[float(value) for value in row]])

    summary_path = output_dir / f"stage1_{gate}_figure_summary.json"
    summary = {
        "gate": gate,
        "checkpoint": data["checkpoint"],
        "pulse_figure": str(pulse_path),
        "phase_figure": str(phase_path),
        "phase_csv": str(csv_path),
        "final_coordinates": {
            name: float(coordinates[-1, index]) for index, name in enumerate(NAMES)
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"pulse": str(pulse_path), "phase": str(phase_path), "summary": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("zzz", "xzz"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("Bilder"))
    parser.add_argument("--steps-per-ns", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data = phase_trajectory(
        root,
        args.checkpoint,
        gate=args.gate,
        steps_per_ns=args.steps_per_ns,
    )
    outputs = save_figures(data, resolve(root, args.output_dir))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()