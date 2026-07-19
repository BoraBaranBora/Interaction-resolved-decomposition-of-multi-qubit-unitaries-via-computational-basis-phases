"""Summarize a saved optimized ZZZ or XZZ checkpoint without rerunning control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from control_optimization.config import ObjectiveWeights
from control_optimization.objective import SupportSelectiveObjective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--gate", choices=["zzz", "xzz"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.result_dir / "pulse_solution.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gate = (args.gate or str(checkpoint.get("gate", "zzz"))).lower()

    full_path = args.result_dir / "optimized_propagator.pt"
    projected_path = args.result_dir / "propagator_projected.pt"
    if full_path.exists():
        propagator = torch.load(full_path, map_location="cpu").to(torch.complex128)
        basis_indices = list(checkpoint["basis_indices"])
    elif projected_path.exists():
        propagator = torch.load(projected_path, map_location="cpu").to(torch.complex128)
        basis_indices = list(range(8))
    else:
        raise FileNotFoundError(
            f"Neither {full_path.name} nor {projected_path.name} exists in {args.result_dir}."
        )

    zero_weights = ObjectiveWeights(
        corrected_infidelity=0.0,
        selected_phase=0.0,
        diagonality=0.0,
        unitarity=0.0,
        survival=0.0,
        fluence=0.0,
        smoothness=0.0,
    )
    objective = SupportSelectiveObjective(
        basis_indices=basis_indices,
        gate=gate,
        target_angle=float(
            checkpoint.get("optimization_config", {}).get(
                "target_angle_rad", math.pi / 4
            )
        ),
        pair_weight=0.2,
        tripartite_weight=0.5,
        weights=zero_weights,
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )
    zero = torch.zeros((), dtype=torch.float64)
    result = objective(propagator, fluence=zero, smoothness=zero)
    summary = {
        "gate": gate.upper(),
        "result_dir": str(args.result_dir),
        "metrics": {k: float(v.detach()) for k, v in result.components.items()},
        "coordinates": {k: float(v.detach()) for k, v in result.coordinates.items()},
    }
    output = args.result_dir / "evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
