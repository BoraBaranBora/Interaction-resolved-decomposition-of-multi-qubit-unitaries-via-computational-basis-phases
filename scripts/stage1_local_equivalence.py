"""Stage 1: reproduce the local-equivalence tripartite-gate optima."""

from __future__ import annotations

import argparse
import copy
import shutil

from pipeline_common import ROOT, load_json, require, run_module, run_python, write_json

INITIAL = {
    "zzz": "results/control_zzz_direct",
    "xzz": "results_paper/pulse_nondiagonal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--gate", choices=("all", "zzz", "xzz"), default="all")
    parser.add_argument("--perturbation-std", type=float, default=2.0e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def result_dir(mode: str, gate: str) -> str:
    prefix = "results_smoke" if mode == "smoke" else "results"
    return f"{prefix}/stage1_local_equivalence/{gate}"


def main() -> None:
    args = parse_args()
    gates = ("zzz", "xzz") if args.gate == "all" else (args.gate,)
    require(["optimize_control.py", "scripts/pipeline_common.py"])

    for gate in gates:
        suffix = "_smoke" if args.mode == "smoke" else ""
        base_path = ROOT / f"configs/control_{gate}_noise_robust{suffix}.json"
        require([base_path, f"{INITIAL[gate]}/pulse_solution.pt"])
        config = copy.deepcopy(load_json(base_path))
        output = result_dir(args.mode, gate)
        output_path = ROOT / output
        if output_path.exists() and args.overwrite:
            shutil.rmtree(output_path)
        config["resume_from"] = INITIAL[gate]
        config["output_dir"] = output
        config["objective_weights"]["population_100_sum"] = 0.0
        config.setdefault("warm_start", {})["parameter_noise_std"] = args.perturbation_std
        config_path = ROOT / f"generated/stage1_configs/{args.mode}_{gate}.json"
        write_json(config_path, config)

        print(f"\n=== Stage 1: {gate.upper()} local-equivalence optimization ===", flush=True)
        run_python("optimize_control.py", "--config", str(config_path))
        figure_steps = "0.02" if args.mode == "smoke" else "1.0"
        run_module(
            "evaluation.stage1_figures",
            "--gate", gate,
            "--checkpoint", output,
            "--steps-per-ns", figure_steps,
            "--output-dir", "Bilder",
        )


if __name__ == "__main__":
    main()
