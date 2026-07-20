"""Stage 3: evaluate local-equivalence and population-refined pulses under OU noise."""

from __future__ import annotations

import argparse
import copy

from pipeline_common import ROOT, load_json, require, run_module, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "overnight", "publication"), default="smoke")
    return parser.parse_args()


def result_prefix(mode: str) -> str:
    return "results_smoke" if mode == "smoke" else "results"


def templates(mode: str) -> tuple[str, str, int, int]:
    if mode == "smoke":
        return "smoke", "smoke", 4, 8
    if mode == "overnight":
        return "overnight", "publication", 64, 256
    return "publication", "publication", 256, 512


def patch_gates(config: dict, *, prefix: str, stage: int, variant: str) -> None:
    config["gates"]["diagonal"]["pulse_dir"] = f"{prefix}/stage{stage}_{'local_equivalence' if stage == 1 else 'population_refinement'}/zzz"
    config["gates"]["nondiagonal"]["pulse_dir"] = f"{prefix}/stage{stage}_{'local_equivalence' if stage == 1 else 'population_refinement'}/xzz"
    for gate in config["gates"].values():
        gate["pulse_variant"] = variant
        gate["note"] = (
            "Stage-one local-equivalence checkpoint."
            if stage == 1
            else "Stage-two population-refined checkpoint."
        )


def write_config(template: str, name: str, *, prefix: str, stage: int, variant: str, n_realizations: int) -> str:
    source = ROOT / template
    config = copy.deepcopy(load_json(source))
    config["name"] = name
    config["n_realizations"] = n_realizations
    patch_gates(config, prefix=prefix, stage=stage, variant=variant)
    destination = ROOT / f"generated/stage3_configs/{name}.json"
    write_json(destination, config)
    return str(destination)


def main() -> None:
    args = parse_args()
    prefix = result_prefix(args.mode)
    require([
        f"{prefix}/stage1_local_equivalence/zzz/pulse_solution.pt",
        f"{prefix}/stage1_local_equivalence/xzz/pulse_solution.pt",
        f"{prefix}/stage2_population_refinement/zzz/pulse_solution.pt",
        f"{prefix}/stage2_population_refinement/xzz/pulse_solution.pt",
    ])
    grid_mode, scenario_mode, grid_n, scenario_n = templates(args.mode)

    local_grid_name = f"stage3_cartesian_local_{args.mode}"
    pop_grid_name = f"stage3_cartesian_population_{args.mode}"
    local_scenario_name = f"stage3_experimental_local_{args.mode}"
    pop_scenario_name = f"stage3_experimental_population_{args.mode}"

    local_grid = write_config(
        f"configs/ou_cartesian_reference_{grid_mode}.json",
        local_grid_name,
        prefix=prefix,
        stage=1,
        variant="local_equivalence",
        n_realizations=grid_n,
    )
    pop_grid = write_config(
        f"configs/ou_cartesian_robust_{grid_mode}.json",
        pop_grid_name,
        prefix=prefix,
        stage=2,
        variant="population_refined",
        n_realizations=grid_n,
    )
    local_scenario = write_config(
        f"configs/ou_experimental_reference_{scenario_mode}.json",
        local_scenario_name,
        prefix=prefix,
        stage=1,
        variant="local_equivalence",
        n_realizations=scenario_n,
    )
    pop_scenario = write_config(
        f"configs/ou_experimental_robust_{scenario_mode}.json",
        pop_scenario_name,
        prefix=prefix,
        stage=2,
        variant="population_refined",
        n_realizations=scenario_n,
    )

    run_module("noise.sweep", "--config", local_grid)
    run_module("noise.sweep", "--config", pop_grid)
    run_module("noise.sweep_scenarios", "--config", local_scenario)
    run_module("noise.sweep_scenarios", "--config", pop_scenario)

    run_module(
        "noise.aggregate_stage3",
        "--local-grid-root", f"results_ou/{local_grid_name}_N{grid_n}",
        "--population-grid-root", f"results_ou/{pop_grid_name}_N{grid_n}",
        "--local-scenario-root", f"results_ou/{local_scenario_name}_N{scenario_n}",
        "--population-scenario-root", f"results_ou/{pop_scenario_name}_N{scenario_n}",
        "--figure-dir", "Bilder",
        "--table-dir", "generated",
    )


if __name__ == "__main__":
    main()
