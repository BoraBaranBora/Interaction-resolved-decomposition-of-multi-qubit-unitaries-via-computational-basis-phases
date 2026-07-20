"""Create the compact stage-one/stage-two comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def load_summary(path: Path) -> dict[str, Any]:
    summary = path / "optimization_summary.json"
    if not summary.exists():
        raise FileNotFoundError(summary)
    return json.loads(summary.read_text(encoding="utf-8"))


def row(gate: str, before: Path, after: Path) -> dict[str, Any]:
    pre = load_summary(before)
    post = load_summary(after)
    p0 = pre["metrics"]
    p1 = post["metrics"]
    return {
        "gate": gate.upper(),
        "local_equivalence_fidelity": float(p0["corrected_fidelity"]),
        "population_refined_fidelity": float(p1["corrected_fidelity"]),
        "fidelity_change": float(p1["corrected_fidelity"] - p0["corrected_fidelity"]),
        "local_equivalence_p100_sum": float(p0["population_100_sum"]),
        "population_refined_p100_sum": float(p1["population_100_sum"]),
        "p100_sum_reduction": float(p0["population_100_sum"] - p1["population_100_sum"]),
        "p100_fractional_reduction": float(
            (p0["population_100_sum"] - p1["population_100_sum"])
            / max(abs(float(p0["population_100_sum"])), 1.0e-15)
        ),
    }


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Noiseless comparison before and after population refinement.}",
        r"\label{tab:population_refinement}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Gate & $F_{\rm local}$ & $F_{\rm pop}$ & $S_{100}^{\rm local}$ & $S_{100}^{\rm pop}$ \\",
        r"\midrule",
    ]
    for item in rows:
        lines.append(
            f"{item['gate']} & {item['local_equivalence_fidelity']:.8f} & "
            f"{item['population_refined_fidelity']:.8f} & "
            f"{item['local_equivalence_p100_sum']:.5g} & "
            f"{item['population_refined_p100_sum']:.5g} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zzz-before", type=Path, required=True)
    parser.add_argument("--zzz-after", type=Path, required=True)
    parser.add_argument("--xzz-before", type=Path, required=True)
    parser.add_argument("--xzz-after", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [
        row("zzz", resolve(root, args.zzz_before), resolve(root, args.zzz_after)),
        row("xzz", resolve(root, args.xzz_before), resolve(root, args.xzz_after)),
    ]
    output = resolve(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "stage2_population_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "stage2_population_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_latex(rows, output / "stage2_population_comparison.tex")
    print(f"Comparison outputs: {output}")


if __name__ == "__main__":
    main()
