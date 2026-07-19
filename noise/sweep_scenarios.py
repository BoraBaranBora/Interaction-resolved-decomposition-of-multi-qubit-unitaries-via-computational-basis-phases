"""Run the three experimentally anchored NV material scenarios for one pulse set.

Each material setting is propagated for both tripartite gates. Completed runs
are skipped unless ``--overwrite`` is supplied, so interrupted production
runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scenarios import MaterialScenario, load_scenarios


GATE_LABELS = {"diagonal": "ZZZ", "nondiagonal": "XZZ"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name", "gates", "noise_scenarios", "n_realizations", "seed",
        "propagation_steps_per_ns",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    if not isinstance(config["gates"], dict) or not config["gates"]:
        raise ValueError("gates must be a nonempty mapping")
    unsupported = sorted(set(config["gates"]).difference(GATE_LABELS))
    if unsupported:
        raise ValueError(f"Unsupported gate keys: {', '.join(unsupported)}")
    for gate, details in config["gates"].items():
        if not isinstance(details, dict) or not details.get("pulse_dir"):
            raise ValueError(f"Gate {gate} requires pulse_dir")
    if int(config["n_realizations"]) <= 0:
        raise ValueError("n_realizations must be positive")
    scenarios = load_scenarios(config["noise_scenarios"])
    config["noise_scenarios"] = [
        {**dict(config["noise_scenarios"][index]), "resolved": scenario.resolved_metadata()["resolved_ou"]}
        for index, scenario in enumerate(scenarios)
    ]
    return config


def git_value(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, check=True,
            capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("numpy", "torch", "matplotlib", "pytest"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_and_tee(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-realizations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else (project_root / args.config).resolve()
    config = load_config(config_path)
    if args.n_realizations is not None:
        if args.n_realizations <= 0:
            raise ValueError("--n-realizations must be positive")
        config["n_realizations"] = args.n_realizations
    if args.seed is not None:
        config["seed"] = args.seed

    n_realizations = int(config["n_realizations"])
    seed = int(config["seed"])
    output_root = args.output_root
    if output_root is None:
        output_root = project_root / "results_ou" / f"{config['name']}_N{n_realizations}"
    elif not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "created_utc": utc_now(),
        "completed_utc": None,
        "config_file": str(config_path),
        "config": config,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": git_value(project_root, "rev-parse", "HEAD"),
            "git_dirty": bool(git_value(project_root, "status", "--porcelain")),
            "packages": package_versions(),
        },
        "runs": [],
    }
    manifest_path = output_root / "sweep_manifest.json"
    total = len(config["gates"]) * len(config["noise_scenarios"])
    print("Experimentally anchored three-material OU study")
    print("Hamiltonian noise term: beta(t) Z_A/2")
    print(f"Output root: {output_root}")
    print(f"Conditions: {total} (2 gates x 3 materials)")
    print(f"Realizations per condition: {n_realizations}")

    run_index = 0
    for gate, gate_details in config["gates"].items():
        gate_label = GATE_LABELS[gate]
        for scenario_payload in config["noise_scenarios"]:
            scenario_input = dict(scenario_payload)
            scenario_input.pop("resolved", None)
            scenario = MaterialScenario.from_dict(scenario_input)
            run_index += 1
            run_name = f"{gate_label}_{scenario.id}"
            run_dir = output_root / run_name
            summary_path = run_dir / "ensemble_summary.json"
            log_path = run_dir / "run.log"
            command = [
                sys.executable, "-m", "noise.run_scenario",
                "--gate", gate,
                "--pulse-dir", str(gate_details["pulse_dir"]),
                "--scenario-json", json.dumps(scenario_input, separators=(",", ":")),
                "--n-realizations", str(n_realizations),
                "--steps-per-ns", str(float(config["propagation_steps_per_ns"])),
                "--seed", str(seed),
                "--output-dir", str(run_dir),
            ]
            record: dict[str, Any] = {
                "index": run_index,
                "gate": gate,
                "gate_label": gate_label,
                "pulse_dir": gate_details["pulse_dir"],
                "scenario": scenario.resolved_metadata(),
                "output_dir": str(run_dir),
                "command": command,
                "status": None,
                "started_utc": utc_now(),
                "finished_utc": None,
                "return_code": None,
            }
            manifest["runs"].append(record)
            print(f"\n[{run_index}/{total}] {run_name}: {scenario.label}")
            if summary_path.exists() and not args.overwrite:
                print("  complete; skipping")
                record.update(status="skipped_complete", return_code=0, finished_utc=utc_now())
                write_json_atomic(manifest_path, manifest)
                continue
            print("  " + subprocess.list2cmdline(command))
            if args.dry_run:
                record.update(status="dry_run", return_code=0, finished_utc=utc_now())
                write_json_atomic(manifest_path, manifest)
                continue
            return_code = run_and_tee(command, cwd=project_root, log_path=log_path)
            record["return_code"] = return_code
            record["finished_utc"] = utc_now()
            record["status"] = "completed" if return_code == 0 and summary_path.exists() else "failed"
            write_json_atomic(manifest_path, manifest)
            if record["status"] == "failed":
                raise RuntimeError(f"Sweep stopped after {run_name}; see {log_path}")

    manifest["completed_utc"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    print(f"\nStudy complete: {output_root}")


if __name__ == "__main__":
    main()
