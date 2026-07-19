"""Compare reference and robust pulses through electron-trajectory diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from control_optimization.trajectory import propagate_with_electron_metrics
from noise.run import (
    checkpoint_path_from,
    load_checkpoint,
    resample_controls,
    validate_checkpoint,
)


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def evaluate_checkpoint(
    project_root: Path,
    checkpoint_value: Path,
    *,
    steps_per_ns: float,
) -> dict[str, Any]:
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from evolution import get_propagator  # type: ignore
    from quantum_model_NV import (  # type: ignore
        get_U_RWA,
        get_precomp,
        qz,
        set_active_carbons,
        ω1,
    )

    checkpoint_path = checkpoint_path_from(resolve(project_root, checkpoint_value))
    checkpoint = load_checkpoint(checkpoint_path)
    original_grid, original_drive, basis_indices, delta_e = validate_checkpoint(
        checkpoint, checkpoint_path
    )
    time_grid, drive = resample_controls(
        original_grid, original_drive, steps_per_ns=steps_per_ns
    )
    set_active_carbons([1, 2])
    precomputed = get_precomp()
    dim_nuc = 3 * (2 ** int(precomputed["N_C"]))
    electron_z = torch.kron(
        torch.as_tensor(qz, dtype=torch.complex128),
        torch.eye(dim_nuc, dtype=torch.complex128),
    )

    def get_step(controls: list[torch.Tensor], dt: float, t: float) -> torch.Tensor:
        return get_U_RWA(controls, dt, t, Δ_e=delta_e, ω_RF=ω1)

    with torch.no_grad():
        reference_propagator = get_propagator(get_step, time_grid, drive)
        metrics = propagate_with_electron_metrics(
            get_step,
            time_grid,
            drive,
            basis_indices=basis_indices,
            electron_z=electron_z,
            sample_stride=1,
            return_traces=True,
        )
        relative = torch.linalg.matrix_norm(
            metrics.propagator - reference_propagator
        ) / torch.clamp(torch.linalg.matrix_norm(reference_propagator), min=1.0e-15)
        propagation_error = float(relative.detach().cpu())
        if propagation_error > 1.0e-8:
            raise RuntimeError(
                "Trajectory propagation does not match evolution.get_propagator: "
                f"relative_error={propagation_error:.3e}."
            )
    return {
        "checkpoint": str(checkpoint_path),
        "time_us": metrics.times.detach().cpu().numpy() * 1e6,
        "exposure_trace": metrics.exposure_trace.detach().cpu().numpy(),
        "excursion_trace": metrics.excursion_trace.detach().cpu().numpy(),
        "excursion_min": metrics.excursion_min_trace.detach().cpu().numpy(),
        "excursion_max": metrics.excursion_max_trace.detach().cpu().numpy(),
        "exposure_integral": float(metrics.dephasing_exposure.detach().cpu()),
        "excursion_integral": float(metrics.manifold_excursion.detach().cpu()),
        "propagator_relative_error": propagation_error,
    }


def plot(reference: dict[str, Any], robust: dict[str, Any], *, gate: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.2), sharex=True)
    for data, label in ((reference, "Reference pulse"), (robust, "Excursion-penalized pulse")):
        line = axes[0].plot(
            data["time_us"],
            data["excursion_trace"],
            label=f"{label}: $\\overline{{I}}_p={data['excursion_integral']:.3f}$",
        )[0]
        axes[0].fill_between(
            data["time_us"],
            data["excursion_min"],
            data["excursion_max"],
            alpha=0.12,
            color=line.get_color(),
        )
        axes[1].plot(
            data["time_us"],
            data["exposure_trace"],
            label=f"{label}: $\\overline{{I}}_Z={data['exposure_integral']:.3f}$",
        )

    axes[0].set_ylabel("Opposite-manifold population")
    axes[0].set_title(f"{gate.upper()} electron-manifold excursion")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Time ($\\mu$s)")
    axes[1].set_ylabel(r"Mean $\mathrm{Var}(Z_A)$")
    axes[1].set_title("Longitudinal dephasing-exposure functional")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("zzz", "xzz"), required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--robust-dir", type=Path, required=True)
    parser.add_argument("--steps-per-ns", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    reference = evaluate_checkpoint(
        project_root, args.reference_dir, steps_per_ns=args.steps_per_ns
    )
    robust = evaluate_checkpoint(
        project_root, args.robust_dir, steps_per_ns=args.steps_per_ns
    )
    output = resolve(project_root, args.output)
    plot(reference, robust, gate=args.gate, output=output)
    summary = {
        "gate": args.gate.upper(),
        "reference": {
            key: value for key, value in reference.items() if not isinstance(value, np.ndarray)
        },
        "robust": {
            key: value for key, value in robust.items() if not isinstance(value, np.ndarray)
        },
        "relative_exposure_reduction": 1.0
        - robust["exposure_integral"] / max(reference["exposure_integral"], 1e-15),
        "relative_excursion_reduction": 1.0
        - robust["excursion_integral"] / max(reference["excursion_integral"], 1e-15),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Figure: {output}")
    print(f"Summary: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
