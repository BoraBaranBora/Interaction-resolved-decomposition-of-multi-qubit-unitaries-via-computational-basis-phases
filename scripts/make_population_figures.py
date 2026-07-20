from __future__ import annotations

import argparse

from pipeline_common import ROOT, require, run_module


CHECKPOINTS = {
    "zzz": "results/control_zzz_noise_robust",
    "xzz": "results/control_xzz_noise_robust",
}
OUTPUTS = {
    "zzz": "Bilder/electron_population_integral_ZZZ.png",
    "xzz": "Bilder/electron_population_integral_XZZ.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("all", "zzz", "xzz"), default="all")
    parser.add_argument("--steps-per-ns", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gates = ("zzz", "xzz") if args.gate == "all" else (args.gate,)
    for gate in gates:
        checkpoint = CHECKPOINTS[gate]
        if args.smoke:
            checkpoint = checkpoint.replace("results/", "results_smoke/")
        require([f"{checkpoint}/pulse_solution.pt"])
        run_module(
            "evaluation.population_integral_original",
            "--gate", gate,
            "--checkpoint", checkpoint,
            "--initial-state", "000",
            "--highlight-state", "100",
            "--steps-per-ns", str(args.steps_per_ns),
            "--output", OUTPUTS[gate],
        )


if __name__ == "__main__":
    main()
