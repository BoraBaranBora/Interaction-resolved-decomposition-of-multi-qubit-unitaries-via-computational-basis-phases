from __future__ import annotations

import argparse

from pipeline_common import require, run_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "overnight", "publication"), required=True)
    return parser.parse_args()


def roots(mode: str) -> tuple[list[tuple[str, str]], list[str]]:
    if mode == "smoke":
        commands = [
            ("noise.sweep", "configs/ou_cartesian_reference_smoke.json"),
            ("noise.sweep", "configs/ou_cartesian_robust_smoke.json"),
            ("noise.sweep_scenarios", "configs/ou_experimental_reference_smoke.json"),
            ("noise.sweep_scenarios", "configs/ou_experimental_robust_smoke.json"),
        ]
        aggregate = [
            "--reference-grid-root", "results_ou/electron_ou_cartesian_reference_smoke_N4",
            "--robust-grid-root", "results_ou/electron_ou_cartesian_robust_smoke_N4",
            "--reference-scenario-root", "results_ou/electron_ou_experimental_reference_smoke_N8",
            "--robust-scenario-root", "results_ou/electron_ou_experimental_robust_smoke_N8",
        ]
    elif mode == "overnight":
        commands = [
            ("noise.sweep", "configs/ou_cartesian_reference_overnight.json"),
            ("noise.sweep", "configs/ou_cartesian_robust_overnight.json"),
            ("noise.sweep_scenarios", "configs/ou_experimental_reference_publication.json"),
            ("noise.sweep_scenarios", "configs/ou_experimental_robust_publication.json"),
        ]
        aggregate = [
            "--reference-grid-root", "results_ou/electron_ou_cartesian_reference_overnight_N64",
            "--robust-grid-root", "results_ou/electron_ou_cartesian_robust_overnight_N64",
            "--reference-scenario-root", "results_ou/electron_ou_experimental_reference_publication_N256",
            "--robust-scenario-root", "results_ou/electron_ou_experimental_robust_publication_N256",
        ]
    else:
        commands = [
            ("noise.sweep", "configs/ou_cartesian_reference_publication.json"),
            ("noise.sweep", "configs/ou_cartesian_robust_publication.json"),
            ("noise.sweep_scenarios", "configs/ou_experimental_reference_publication.json"),
            ("noise.sweep_scenarios", "configs/ou_experimental_robust_publication.json"),
        ]
        aggregate = [
            "--reference-grid-root", "results_ou/electron_ou_cartesian_reference_publication_N256",
            "--robust-grid-root", "results_ou/electron_ou_cartesian_robust_publication_N256",
            "--reference-scenario-root", "results_ou/electron_ou_experimental_reference_publication_N512",
            "--robust-scenario-root", "results_ou/electron_ou_experimental_robust_publication_N512",
        ]
    return commands, aggregate


def main() -> None:
    args = parse_args()
    robust_prefix = "results_smoke" if args.mode == "smoke" else "results"
    require([
        "results/control_zzz_direct/pulse_solution.pt",
        "results_paper/pulse_nondiagonal/pulse_solution.pt",
        f"{robust_prefix}/control_zzz_noise_robust/pulse_solution.pt",
        f"{robust_prefix}/control_xzz_noise_robust/pulse_solution.pt",
    ])
    commands, aggregate = roots(args.mode)
    for module, config in commands:
        run_module(module, "--config", config)
    run_module(
        "noise.aggregate_comparison",
        *aggregate,
        "--figure-dir", "Bilder",
        "--table-dir", "generated",
    )


if __name__ == "__main__":
    main()
