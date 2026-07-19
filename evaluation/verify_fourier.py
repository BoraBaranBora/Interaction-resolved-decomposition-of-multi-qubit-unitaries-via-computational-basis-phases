from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from control_optimization.pulse import DirectFourierPulse, FourierPulseBounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a saved control checkpoint is exactly one direct Fourier "
            "pulse of the form used in the manuscript."
        )
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--peak-tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def verify(result_dir: Path, tolerance: float, peak_tolerance: float) -> dict[str, float | int | bool]:
    checkpoint_path = result_dir / "pulse_solution.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameterization = str(checkpoint.get("pulse_parameterization", ""))
    if parameterization != "direct_fourier":
        raise ValueError(
            f"Checkpoint parameterization is {parameterization!r}, not 'direct_fourier'."
        )

    settings = checkpoint["pulse_settings"][0]
    bounds = FourierPulseBounds(
        basis_size=int(settings["basis_size"]),
        max_field_uT=float(settings["maximal_pulse"]),
        min_angular_frequency=float(settings["minimal_frequency"]),
        max_angular_frequency=float(settings["maximal_frequency"]),
        phase_bound=float(settings["maximal_phase"]),
        taper_fraction=float(settings["taper_fraction"]),
    )
    model = DirectFourierPulse(bounds)
    physical = torch.as_tensor(checkpoint["params"], dtype=torch.float64).reshape(-1)
    time_grid = torch.as_tensor(checkpoint["time_grid"], dtype=torch.float64).reshape(-1)
    saved_drive = torch.as_tensor(checkpoint["drive"][0], dtype=torch.float64).reshape(-1)
    raw = model.raw_from_physical(physical)
    reconstructed = model.drive(time_grid, raw)

    relative_error = float(
        torch.linalg.vector_norm(reconstructed - saved_drive)
        / torch.clamp(torch.linalg.vector_norm(saved_drive), min=1.0e-30)
    )
    peak_uT = float(saved_drive.abs().max())
    peak_fraction = peak_uT / bounds.max_field_uT
    endpoint_max_uT = max(abs(float(saved_drive[0])), abs(float(saved_drive[-1])))
    exact = relative_error <= tolerance
    peak_ok = peak_fraction <= 1.0 + peak_tolerance

    return {
        "basis_size": bounds.basis_size,
        "relative_reconstruction_error": relative_error,
        "peak_field_uT": peak_uT,
        "peak_bound_uT": bounds.max_field_uT,
        "peak_fraction_of_bound": peak_fraction,
        "endpoint_max_uT": endpoint_max_uT,
        "equation_exact": exact,
        "peak_constraint_satisfied": peak_ok,
    }


def main() -> None:
    args = parse_args()
    report = verify(args.result_dir, args.tolerance, args.peak_tolerance)
    print(f"Direct Fourier basis size: {report['basis_size']}")
    print(
        "Coefficient-to-drive reconstruction error: "
        f"{report['relative_reconstruction_error']:.3e}"
    )
    print(
        f"Sampled peak: {report['peak_field_uT']:.6f} uT / "
        f"{report['peak_bound_uT']:.6f} uT "
        f"({report['peak_fraction_of_bound']:.6f})"
    )
    print(f"Endpoint magnitude: {report['endpoint_max_uT']:.3e} uT")
    print(f"Manuscript equation exact: {report['equation_exact']}")
    print(f"Peak constraint satisfied: {report['peak_constraint_satisfied']}")
    if not report["equation_exact"] or not report["peak_constraint_satisfied"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
