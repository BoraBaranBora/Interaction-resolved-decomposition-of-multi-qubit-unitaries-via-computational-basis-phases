"""Propagate a saved NV tripartite gate under electronic OU detuning noise.

The production model is deliberately narrow:

    H_r(t) = H_NV(t) + beta_r(t) * (Z_A / 2).

The stochastic term is inserted inside every short-time propagator. Each run
uses one source-native material scenario resolved into a common OU pair
(sigma_A, tau_c). Noise-spectroscopy or echo data may be used to infer those
environmental parameters, but no echo, dynamical-decoupling, or nuclear-noise
channel is inserted into the gate simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import (
    corrected_logical_operator,
    electronic_dephasing_operator,
    fixed_local_correction,
    frame_for_gate,
    logical_block,
    logical_gate_metrics,
    target_in_diagonal_frame,
)
from .ou import sample_ou_process
from .scenarios import MaterialScenario


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (project_root / path).resolve()


def checkpoint_path_from(value: Path) -> Path:
    if value.is_dir():
        value = value / "pulse_solution.pt"
    if not value.exists():
        raise FileNotFoundError(f"Could not find pulse checkpoint: {value}")
    return value


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("pulse_solution.pt must contain a dictionary")
    required = ("time_grid", "drive", "basis_indices")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing: {', '.join(missing)}")
    return checkpoint


def checkpoint_delta_e(checkpoint: dict[str, Any]) -> float:
    for key in ("Δ", "Δ_e", "delta_e"):
        if key in checkpoint:
            return float(checkpoint[key])
    raise KeyError("Checkpoint contains no electron detuning key (Δ, Δ_e, delta_e)")


def validate_checkpoint(
    checkpoint: dict[str, Any], checkpoint_path: Path
) -> tuple[torch.Tensor, list[torch.Tensor], list[int], float]:
    time_grid = torch.as_tensor(checkpoint["time_grid"], dtype=torch.float64)
    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("checkpoint time_grid must be one-dimensional")
    if torch.any(time_grid[1:] <= time_grid[:-1]):
        raise ValueError("checkpoint time_grid must be strictly increasing")

    raw_drive = checkpoint["drive"]
    if isinstance(raw_drive, torch.Tensor):
        raw_drive = [raw_drive]
    if not isinstance(raw_drive, (list, tuple)) or not raw_drive:
        raise ValueError("checkpoint drive must contain at least one channel")
    drive = [torch.as_tensor(channel, dtype=torch.float64) for channel in raw_drive]
    for channel in drive:
        if channel.ndim != 1 or channel.numel() < time_grid.numel() - 1:
            raise ValueError(
                f"Drive in {checkpoint_path} is shorter than the propagation grid"
            )

    basis_indices = [int(value) for value in checkpoint["basis_indices"]]
    if len(basis_indices) != 8 or len(set(basis_indices)) != 8:
        raise ValueError("checkpoint basis_indices must contain 8 unique indices")

    return time_grid, drive, basis_indices, checkpoint_delta_e(checkpoint)



def resample_controls(
    time_grid: torch.Tensor,
    drive: list[torch.Tensor],
    *,
    steps_per_ns: float | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Linearly resample all controls to a common uniform propagation grid."""
    if steps_per_ns is None:
        return time_grid, drive
    resolution = float(steps_per_ns)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("steps_per_ns must be finite and positive")

    start = float(time_grid[0].item())
    stop = float(time_grid[-1].item())
    duration_ns = (stop - start) * 1e9
    n_intervals = max(1, int(round(duration_ns * resolution)))
    new_grid = torch.linspace(start, stop, n_intervals + 1, dtype=torch.float64)
    old_x = time_grid.detach().cpu().numpy()
    new_x = new_grid.detach().cpu().numpy()
    resampled: list[torch.Tensor] = []
    for channel in drive:
        values = channel.detach().cpu()
        if values.numel() == time_grid.numel() - 1:
            values = torch.cat((values, values[-1:]))
        else:
            values = values[: time_grid.numel()]
        resampled.append(
            torch.from_numpy(np.interp(new_x, old_x, values.numpy())).to(torch.float64)
        )
    return new_grid, resampled


def propagate(
    *,
    get_u: Any,
    time_grid: torch.Tensor,
    drive: list[torch.Tensor],
    delta_e: float,
    omega_rf: Any,
    electron_z_half: torch.Tensor,
    beta_trace: torch.Tensor,
) -> torch.Tensor:
    """Propagate one realization with beta_n Z_A/2 inside every step."""
    beta_trace = torch.as_tensor(beta_trace, dtype=torch.float64)
    if beta_trace.shape != time_grid.shape:
        raise ValueError("beta_trace and time_grid must have the same shape")

    dimension = electron_z_half.shape[0]
    propagator = torch.eye(dimension, dtype=torch.complex128)
    for index in range(time_grid.numel() - 1):
        dt = float((time_grid[index + 1] - time_grid[index]).item())
        time = float(time_grid[index].item())
        controls = [channel[index] for channel in drive]
        h_noise = beta_trace[index] * electron_z_half
        step = get_u(
            controls,
            dt,
            time,
            Δ_e=delta_e,
            ω_RF=omega_rf,
            H_noise=h_noise,
        )
        propagator = step @ propagator
    return propagator


def write_outputs(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    f_ent = np.asarray(metrics["entanglement_fidelity_samples"], dtype=float)
    f_avg = np.asarray(metrics["average_gate_fidelity_samples"], dtype=float)
    survival = np.asarray(metrics["survival_samples"], dtype=float)

    with (output_dir / "ensemble_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "realization",
                "entanglement_fidelity",
                "average_gate_fidelity",
                "logical_survival",
            ]
        )
        for index, values in enumerate(zip(f_ent, f_avg, survival)):
            writer.writerow([index, *values])

    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("diagonal", "nondiagonal"), required=True)
    parser.add_argument(
        "--pulse-dir",
        type=Path,
        required=True,
        help="Directory containing pulse_solution.pt, or the checkpoint itself.",
    )
    parser.add_argument(
        "--scenario-json",
        type=str,
        required=True,
        help="Complete source-native material scenario as a JSON object.",
    )
    parser.add_argument("--n-realizations", type=int, default=256)
    parser.add_argument(
        "--steps-per-ns",
        type=float,
        default=1.0,
        help="Uniform propagation resolution after control interpolation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.n_realizations <= 0:
        raise ValueError("--n-realizations must be positive")

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    sys.path.insert(0, str(src_dir))

    # Repository-specific imports are intentionally lazy, so unit tests for the
    # noise model do not require the full NV simulation stack.
    from quantum_model_NV import (  # type: ignore
        get_U_RWA,
        get_precomp,
        qz,
        set_active_carbons,
        ω1,
    )

    pulse_value = resolve_path(project_root, args.pulse_dir)
    checkpoint_path = checkpoint_path_from(pulse_value)
    output_dir = resolve_path(project_root, args.output_dir)
    checkpoint = load_checkpoint(checkpoint_path)
    original_time_grid, original_drive, basis_indices, delta_e = validate_checkpoint(
        checkpoint, checkpoint_path
    )
    time_grid, drive = resample_controls(
        original_time_grid, original_drive, steps_per_ns=args.steps_per_ns
    )

    set_active_carbons([1, 2])
    precomputed = get_precomp()
    dim_nuc = int(precomputed["dim_nuc"])
    electron_z_half = electronic_dephasing_operator(qz, dim_nuc)
    expected_dimension = 2 * dim_nuc
    if max(basis_indices) >= expected_dimension:
        raise ValueError("checkpoint logical basis is incompatible with NV model dimension")

    scenario = MaterialScenario.from_dict(json.loads(args.scenario_json))
    params = scenario.ou_parameters()
    scenario_metadata = scenario.resolved_metadata()
    traces = sample_ou_process(
        time_grid,
        params,
        n_realizations=args.n_realizations,
        seed=args.seed,
    )

    frame = frame_for_gate(args.gate)
    target = target_in_diagonal_frame()
    zero_trace = torch.zeros_like(time_grid)

    print("Noise model: H_r(t) = H_NV(t) + beta_r(t) * Z_A/2")
    print("Calibration: source-native material scenario")
    print(f"Pulse checkpoint: {checkpoint_path}")
    print(f"Gate: {args.gate}")
    print(f"Scenario: {scenario.label} [{scenario.id}]")
    print(f"Equivalent T2*: {params.equivalent_t2_star_s * 1e6:.6g} us")
    print(f"tau_c: {params.correlation_time_s * 1e6:.6g} us")
    print(f"sigma/2pi: {params.sigma_rad_s / (2.0 * math.pi * 1e3):.6g} kHz")
    print(f"Realizations: {args.n_realizations}")
    print(f"Propagation resolution: {args.steps_per_ns:g} steps/ns")

    with torch.no_grad():
        nominal_full = propagate(
            get_u=get_U_RWA,
            time_grid=time_grid,
            drive=drive,
            delta_e=delta_e,
            omega_rf=ω1,
            electron_z_half=electron_z_half,
            beta_trace=zero_trace,
        )
        nominal_logical = logical_block(nominal_full, basis_indices)
        nominal_framed = frame @ nominal_logical @ frame.conj().T
        correction, local_coordinates = fixed_local_correction(nominal_framed)
        nominal_corrected = correction @ nominal_framed
        noiseless_metrics = logical_gate_metrics(target, [nominal_corrected])

        corrected_realizations: list[torch.Tensor] = []
        progress_step = max(1, args.n_realizations // 10)
        for realization in range(args.n_realizations):
            if realization % progress_step == 0:
                print(f"  realization {realization}/{args.n_realizations}")
            full = propagate(
                get_u=get_U_RWA,
                time_grid=time_grid,
                drive=drive,
                delta_e=delta_e,
                omega_rf=ω1,
                electron_z_half=electron_z_half,
                beta_trace=traces[realization],
            )
            corrected_realizations.append(
                corrected_logical_operator(
                    full,
                    basis_indices=basis_indices,
                    frame=frame,
                    correction=correction,
                )
            )

    metrics = logical_gate_metrics(target, corrected_realizations)
    regime_metadata = scenario_metadata
    q05, q50, q95 = metrics["average_gate_fidelity_quantiles_05_50_95"]
    noiseless_favg = float(noiseless_metrics["average_gate_fidelity_mean"])
    noisy_favg = float(metrics["average_gate_fidelity_mean"])

    summary: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "gate": "ZZZ" if args.gate == "diagonal" else "XZZ",
        "gate_key": args.gate,
        "pulse_checkpoint": str(checkpoint_path),
        "pulse_checkpoint_sha256": sha256(checkpoint_path),
        "basis_indices": basis_indices,
        "n_realizations": args.n_realizations,
        "seed": args.seed,
        "noise_model": {
            "name": "electron_longitudinal_ornstein_uhlenbeck",
            "hamiltonian_term": "beta(t) * (Z_A/2 tensor I_nuclear)",
            "calibration": params.calibration,
            "scenario_id": scenario.id,
            "equivalent_t2_star_us": params.equivalent_t2_star_s * 1e6,
            "correlation_time_us": params.correlation_time_s * 1e6,
            "sigma_rad_s": params.sigma_rad_s,
            "stationary_initial_state": True,
            "exact_discrete_update": True,
            "propagation_steps_per_ns": args.steps_per_ns,
            "original_time_points": int(original_time_grid.numel()),
            "propagation_time_points": int(time_grid.numel()),
            "hahn_echo_or_dd_applied": False,
            "nuclear_noise_channels": False,
        },
        "material_scenario": regime_metadata,
        "fixed_local_correction": {
            key: float(value.detach().cpu())
            for key, value in local_coordinates.items()
            if key in {"A", "B", "C"}
        },
        "noiseless": {
            "entanglement_fidelity": float(
                noiseless_metrics["entanglement_fidelity_mean"]
            ),
            "average_gate_fidelity": noiseless_favg,
            "logical_survival": float(noiseless_metrics["survival_mean"]),
        },
        "entanglement_fidelity_mean": float(
            metrics["entanglement_fidelity_mean"]
        ),
        "entanglement_fidelity_stderr": float(
            metrics["entanglement_fidelity_stderr"]
        ),
        "average_gate_fidelity_mean": noisy_favg,
        "average_gate_fidelity_stderr": float(
            metrics["average_gate_fidelity_stderr"]
        ),
        "survival_mean": float(metrics["survival_mean"]),
        "survival_stderr": float(metrics["survival_stderr"]),
        "average_gate_fidelity_quantiles_05_50_95": [q05, q50, q95],
        "dephasing_induced_average_fidelity_loss": noiseless_favg - noisy_favg,
    }
    write_outputs(output_dir=output_dir, summary=summary, metrics=metrics)

    print(
        "Average gate fidelity = "
        f"{noisy_favg:.10f} +/- "
        f"{metrics['average_gate_fidelity_stderr']:.3e} (SEM)"
    )
    print(f"Quantiles [5%, 50%, 95%] = [{q05:.10f}, {q50:.10f}, {q95:.10f}]")
    print(f"Logical survival = {metrics['survival_mean']:.10f}")
    print(f"Noiseless average gate fidelity = {noiseless_favg:.10f}")
    print(f"Dephasing-induced fidelity loss = {noiseless_favg - noisy_favg:+.10e}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
