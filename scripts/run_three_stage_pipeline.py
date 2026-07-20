"""Run all three manuscript result stages in sequence."""

from __future__ import annotations

import argparse

from pipeline_common import run_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "overnight", "publication"), default="smoke")
    parser.add_argument("--minimum-fidelity", type=float, default=0.997)
    parser.add_argument("--perturbation-std", type=float, default=2.0e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimization_mode = "smoke" if args.mode == "smoke" else "production"
    common = ["--overwrite"] if args.overwrite else []
    run_python(
        "scripts/stage1_local_equivalence.py",
        "--mode", optimization_mode,
        "--gate", "all",
        "--perturbation-std", str(args.perturbation_std),
        *common,
    )
    run_python(
        "scripts/stage2_population_refinement.py",
        "--mode", optimization_mode,
        "--gate", "all",
        "--minimum-fidelity", str(args.minimum_fidelity),
        *common,
    )
    run_python("scripts/stage3_noise_evaluation.py", "--mode", args.mode)


if __name__ == "__main__":
    main()
