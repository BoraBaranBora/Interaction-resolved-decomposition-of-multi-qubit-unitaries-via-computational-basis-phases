"""Select a population-refined pulse while controlling fidelity loss."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--minimum-fidelity", type=float, default=0.997)
    parser.add_argument("--maximum-fidelity-drop", type=float, default=1.0e-3)
    parser.add_argument("--allow-threshold-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_row(root: Path, value: Path) -> dict:
    directory = value if value.is_absolute() else (root / value).resolve()
    summary_path = directory / "optimization_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    return {
        "directory": directory,
        "fidelity": float(metrics["corrected_fidelity"]),
        "population_100_sum": float(metrics["population_100_sum"]),
        "loss": float(summary["loss"]),
    }


def main() -> None:
    args = parse_args()
    if args.maximum_fidelity_drop < 0.0:
        raise ValueError("maximum-fidelity-drop cannot be negative")

    root = Path(__file__).resolve().parents[1]
    rows = [load_row(root, value) for value in args.candidate]

    reference = load_row(root, args.reference) if args.reference is not None else None
    relative_floor = (
        reference["fidelity"] - args.maximum_fidelity_drop
        if reference is not None
        else float("-inf")
    )

    strict = [
        row
        for row in rows
        if row["fidelity"] >= args.minimum_fidelity
        and row["fidelity"] >= relative_floor
    ]

    threshold_met = bool(strict)
    selection_mode = "absolute_and_relative_threshold"
    eligible = strict

    if not eligible:
        details = "\n".join(
            f"{row['directory']}: F={row['fidelity']:.10f}, "
            f"P100 sum={row['population_100_sum']:.6g}"
            for row in rows
        )
        if not args.allow_threshold_fallback:
            reference_text = (
                f"\nReference fidelity={reference['fidelity']:.10f}; "
                f"relative floor={relative_floor:.10f}."
                if reference is not None
                else ""
            )
            raise RuntimeError(
                f"No candidate reached minimum fidelity {args.minimum_fidelity}."
                f"{reference_text}\n{details}"
            )

        eligible = [row for row in rows if row["fidelity"] >= relative_floor]
        if eligible:
            selection_mode = "relative_fidelity_fallback"
            print(
                "[smoke fallback] No candidate met the absolute fidelity floor; "
                f"selecting the smallest P100 sum within {args.maximum_fidelity_drop:g} "
                "of the stage-one fidelity.",
                flush=True,
            )
        else:
            # This should be rare. Keep the run usable, but select the highest-
            # fidelity candidate rather than a low-population, low-fidelity pulse.
            best_fidelity = max(row["fidelity"] for row in rows)
            eligible = [row for row in rows if row["fidelity"] == best_fidelity]
            selection_mode = "highest_fidelity_emergency_fallback"
            print(
                "[smoke fallback] No candidate stayed within the allowed fidelity "
                "drop; selecting the highest-fidelity candidate.",
                flush=True,
            )

    selected = min(
        eligible,
        key=lambda row: (row["population_100_sum"], -row["fidelity"]),
    )

    output = args.output_dir if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    shutil.copytree(selected["directory"], output)

    selection = {
        "minimum_fidelity": args.minimum_fidelity,
        "maximum_fidelity_drop": args.maximum_fidelity_drop,
        "threshold_met": threshold_met,
        "selection_mode": selection_mode,
        "reference": (
            {**reference, "directory": str(reference["directory"])}
            if reference is not None
            else None
        ),
        "selected": {**selected, "directory": str(selected["directory"])},
        "candidates": [
            {**row, "directory": str(row["directory"])} for row in rows
        ],
    }
    (output / "population_refinement_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    print(f"Selection mode: {selection_mode}")
    print(f"Absolute threshold met: {threshold_met}")
    print(f"Selected: {selected['directory']}")
    print(f"Fidelity: {selected['fidelity']:.10f}")
    print(f"P100 sum: {selected['population_100_sum']:.8f}")
    print(f"Population-refined output: {output}")


if __name__ == "__main__":
    main()
