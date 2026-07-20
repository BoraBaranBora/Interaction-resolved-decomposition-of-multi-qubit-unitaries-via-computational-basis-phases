from __future__ import annotations

import argparse

from pipeline_common import require, run_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("all", "zzz", "xzz"), default="all")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require([
        "optimize_control.py",
        "configs/control_zzz_noise_robust_smoke.json",
        "configs/control_xzz_noise_robust_smoke.json",
    ])
    gates = ("zzz", "xzz") if args.gate == "all" else (args.gate,)
    for gate in gates:
        print(f"\n=== {gate.upper()} smoke optimization ===", flush=True)
        run_python(
            "optimize_control.py",
            "--config",
            f"configs/control_{gate}_noise_robust_smoke.json",
        )
    if not args.skip_plots:
        run_python(
            "scripts/make_population_figures.py",
            "--gate", args.gate,
            "--steps-per-ns", "0.02",
            "--smoke",
        )


if __name__ == "__main__":
    main()
