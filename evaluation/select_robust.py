"""Select the lowest-P100-sum pulse satisfying a fidelity threshold."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--minimum-fidelity", type=float, default=0.997)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    for value in args.candidate:
        directory = value if value.is_absolute() else (root / value).resolve()
        summary_path = directory / "optimization_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        rows.append(
            {
                "directory": directory,
                "fidelity": float(metrics["corrected_fidelity"]),
                "population_100_sum": float(metrics["population_100_sum"]),
                "loss": float(summary["loss"]),
            }
        )
    accepted = [row for row in rows if row["fidelity"] >= args.minimum_fidelity]
    if not accepted:
        details = "\n".join(
            f"{row['directory']}: F={row['fidelity']:.10f}, P100 sum={row['population_100_sum']:.6g}"
            for row in rows
        )
        raise RuntimeError(
            f"No candidate reached minimum fidelity {args.minimum_fidelity}.\n{details}"
        )
    selected = min(accepted, key=lambda row: (row["population_100_sum"], -row["fidelity"]))
    output = args.output_dir if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    shutil.copytree(selected["directory"], output)
    selection = {
        "minimum_fidelity": args.minimum_fidelity,
        "selected": {**selected, "directory": str(selected["directory"])},
        "candidates": [{**row, "directory": str(row["directory"])} for row in rows],
    }
    (output / "robust_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(f"Selected: {selected['directory']}")
    print(f"Fidelity: {selected['fidelity']:.10f}")
    print(f"P100 sum: {selected['population_100_sum']:.8f}")
    print(f"Canonical output: {output}")


if __name__ == "__main__":
    main()
