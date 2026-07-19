"""Validate repository integration before launching OU ensembles."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .sweep import load_config


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def checkpoint_file(value: Path) -> Path:
    return value / "pulse_solution.pt" if value.is_dir() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ou_electron_ramsey_grid.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve(project_root, args.config)
    config = load_config(config_path)

    sys.path.insert(0, str(project_root / "src"))
    from quantum_model_NV import get_U_RWA  # type: ignore

    signature = inspect.signature(get_U_RWA)
    if "H_noise" not in signature.parameters:
        raise RuntimeError(
            "quantum_model_NV.get_U_RWA has no H_noise argument. Apply the "
            "Hamiltonian-noise insertion patch before running ensembles."
        )

    print("get_U_RWA accepts H_noise: yes")
    print("Primary model: electron-only Ramsey-calibrated OU detuning")
    print(f"Configuration: {config_path}")

    for gate, details in config["gates"].items():
        path = checkpoint_file(resolve(project_root, details["pulse_dir"]))
        if not path.exists():
            raise FileNotFoundError(f"Missing {gate} checkpoint: {path}")
        checkpoint: dict[str, Any] = torch.load(
            path, map_location="cpu", weights_only=False
        )
        basis_size = None
        settings = checkpoint.get("pulse_settings")
        if isinstance(settings, list) and settings and isinstance(settings[0], dict):
            basis_size = settings[0].get("basis_size")
        parameterization = checkpoint.get("pulse_parameterization", "legacy/unspecified")
        basis_indices = checkpoint.get("basis_indices")
        if basis_indices is None or len(basis_indices) != 8:
            raise RuntimeError(f"{gate} checkpoint has no valid 8-state basis_indices")
        print(
            f"{gate}: {path} | basis_size={basis_size} | "
            f"parameterization={parameterization}"
        )

    t2 = [entry["value_us"] for entry in config["t2_star_regimes"]]
    tau = [entry["value_us"] for entry in config["tau_c_regimes"]]
    print(f"T2* grid (us): {t2}")
    print(f"tau_c grid (us): {tau}")
    print(f"Realizations: {config['n_realizations']}")
    print(f"Propagation resolution: {config['propagation_steps_per_ns']} steps/ns")
    print("Validation passed.")


if __name__ == "__main__":
    main()
