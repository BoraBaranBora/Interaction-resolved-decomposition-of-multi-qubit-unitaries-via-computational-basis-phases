from __future__ import annotations

import argparse
import copy

from pipeline_common import ROOT, load_json, require, run_module, run_python, write_json

WEIGHTS = (1.0e-4, 3.0e-4, 1.0e-3)
INITIAL = {
    "zzz": "results/control_zzz_direct",
    "xzz": "results_paper/pulse_nondiagonal",
}
EXPECTED_PARAMETERIZATION = {
    "zzz": "direct_fourier",
    "xzz": "reference_residual_fourier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("all", "zzz", "xzz"), default="all")
    parser.add_argument("--minimum-fidelity", type=float, default=0.9970)
    parser.add_argument("--overwrite-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gates = ("zzz", "xzz") if args.gate == "all" else (args.gate,)
    require(["optimize_control.py"])

    for gate in gates:
        base_path = ROOT / f"configs/control_{gate}_noise_robust.json"
        require([base_path, f"{INITIAL[gate]}/pulse_solution.pt"])
        base = load_json(base_path)
        actual = str(base.get("pulse_parameterization", "")).lower()
        expected = EXPECTED_PARAMETERIZATION[gate]
        if actual != expected:
            raise ValueError(
                f"{gate.upper()} config must use {expected!r}, found {actual!r}."
            )

        resume = INITIAL[gate]
        candidates: list[str] = []
        for weight in WEIGHTS:
            slug = f"{weight:.2f}".replace(".", "p")
            output_dir = f"results/control_{gate}_noise_robust_w{slug}"
            config = copy.deepcopy(base)
            config["resume_from"] = resume
            config["output_dir"] = output_dir
            config["objective_weights"]["population_100_sum"] = weight
            config_path = ROOT / f"results/_robust_configs/control_{gate}_w{slug}.json"
            write_json(config_path, config)

            print(f"\n=== {gate.upper()} P100-sum weight {weight:g} ===", flush=True)
            run_python("optimize_control.py", "--config", str(config_path))
            candidates.append(output_dir)
            resume = output_dir

        selection = [
            "evaluation.select_robust",
            "--minimum-fidelity", str(args.minimum_fidelity),
            "--output-dir", f"results/control_{gate}_noise_robust",
        ]
        for candidate in candidates:
            selection += ["--candidate", candidate]
        if args.overwrite_selection:
            selection.append("--overwrite")
        print(f"\n=== Select {gate.upper()} robust checkpoint ===", flush=True)
        run_module(*selection)


if __name__ == "__main__":
    main()
