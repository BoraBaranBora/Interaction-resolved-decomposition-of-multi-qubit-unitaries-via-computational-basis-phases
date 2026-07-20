"""Stage 2: refine the stage-one pulses using the sampled P100 sum."""

from __future__ import annotations

import argparse
import copy
import shutil

from pipeline_common import ROOT, load_json, require, run_module, run_python, write_json

DEFAULT_WEIGHTS = (1.0e-4, 3.0e-4, 1.0e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--gate", choices=("all", "zzz", "xzz"), default="all")
    parser.add_argument("--minimum-fidelity", type=float, default=0.997)
    parser.add_argument("--maximum-fidelity-drop", type=float, default=1.0e-3)
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prefix(mode: str) -> str:
    return "results_smoke" if mode == "smoke" else "results"


def stage1_dir(mode: str, gate: str) -> str:
    return f"{prefix(mode)}/stage1_local_equivalence/{gate}"


def stage2_dir(mode: str, gate: str) -> str:
    return f"{prefix(mode)}/stage2_population_refinement/{gate}"


def weight_slug(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def main() -> None:
    args = parse_args()
    gates = ("zzz", "xzz") if args.gate == "all" else (args.gate,)
    require(["optimize_control.py"])

    for gate in gates:
        before = stage1_dir(args.mode, gate)
        require([f"{before}/pulse_solution.pt", f"{before}/optimization_summary.json"])
        suffix = "_smoke" if args.mode == "smoke" else ""
        base_path = ROOT / f"configs/control_{gate}_noise_robust{suffix}.json"
        require([base_path])
        base = load_json(base_path)
        resume = before
        candidates: list[str] = []

        for weight in args.weights:
            output = f"{prefix(args.mode)}/stage2_population_refinement/{gate}_{weight_slug(weight)}"
            candidates.append(output)
            if args.select_only:
                require([f"{output}/pulse_solution.pt", f"{output}/optimization_summary.json"])
                continue

            config = copy.deepcopy(base)
            output_path = ROOT / output
            if output_path.exists() and args.overwrite:
                shutil.rmtree(output_path)
            config["resume_from"] = resume
            config["output_dir"] = output
            config["objective_weights"]["population_100_sum"] = float(weight)
            config.setdefault("warm_start", {})["parameter_noise_std"] = 0.0
            config_path = ROOT / f"generated/stage2_configs/{args.mode}_{gate}_{weight_slug(weight)}.json"
            write_json(config_path, config)

            print(
                f"\n=== Stage 2: {gate.upper()} population refinement, weight={weight:g} ===",
                flush=True,
            )
            run_python("optimize_control.py", "--config", str(config_path))
            resume = output

        canonical = stage2_dir(args.mode, gate)
        selection_args = [
            "evaluation.select_population_refined",
            "--reference", before,
            "--minimum-fidelity", str(args.minimum_fidelity),
            "--maximum-fidelity-drop", str(args.maximum_fidelity_drop),
            "--output-dir", canonical,
        ]
        # Include stage one as a valid no-refinement fallback. This guarantees
        # that stage two never silently returns a pulse worse than the baseline.
        selection_args += ["--candidate", before]
        for candidate in candidates:
            selection_args += ["--candidate", candidate]
        if args.mode == "smoke":
            selection_args.append("--allow-threshold-fallback")
        if args.overwrite:
            selection_args.append("--overwrite")
        run_module(*selection_args)

        plot_steps = "0.02" if args.mode == "smoke" else "1.0"
        run_module(
            "evaluation.population_integral_original",
            "--gate", gate,
            "--checkpoint", before,
            "--initial-state", "000",
            "--highlight-state", "100",
            "--steps-per-ns", plot_steps,
            "--output", f"Bilder/stage2_{gate.upper()}_population_before.png",
        )
        run_module(
            "evaluation.population_integral_original",
            "--gate", gate,
            "--checkpoint", canonical,
            "--initial-state", "000",
            "--highlight-state", "100",
            "--steps-per-ns", plot_steps,
            "--output", f"Bilder/stage2_{gate.upper()}_population_after.png",
        )

    zzz_before = stage1_dir(args.mode, "zzz")
    zzz_after = stage2_dir(args.mode, "zzz")
    xzz_before = stage1_dir(args.mode, "xzz")
    xzz_after = stage2_dir(args.mode, "xzz")
    if all((ROOT / value / "optimization_summary.json").exists() for value in (zzz_before, zzz_after, xzz_before, xzz_after)):
        run_module(
            "evaluation.compare_population_stages",
            "--zzz-before", zzz_before,
            "--zzz-after", zzz_after,
            "--xzz-before", xzz_before,
            "--xzz-after", xzz_after,
            "--output-dir", "generated",
        )


if __name__ == "__main__":
    main()
