"""Aggregate stage-one and stage-two OU runs using the original single-axis plot style."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from .aggregate import load_rows as load_grid_rows


GATES = ("ZZZ", "XZZ")
TAU_VALUES = (0.1, 0.3, 1.0, 3.0, 15.0, 30.0)
SCENARIO_ORDER = ("bargill_12c_cvd", "hayashi_hpht_no8", "bauch_12c_2ppm")
SCENARIO_MARKERS = {
    "bargill_12c_cvd": "D",
    "hayashi_hpht_no8": "*",
    "bauch_12c_2ppm": "P",
}
SCENARIO_SHORT = {
    "bargill_12c_cvd": "Bar-Gill/Walsworth",
    "hayashi_hpht_no8": "Hayashi",
    "bauch_12c_2ppm": "Bauch/Walsworth",
}


def resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def positive(values: np.ndarray | float) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)


def load_scenario_rows(root: Path, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("ensemble_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        scenario = summary["material_scenario"]
        resolved = scenario["resolved_ou"]
        quantiles = summary["average_gate_fidelity_quantiles_05_50_95"]
        noiseless = summary.get("noiseless", {})
        rows.append(
            {
                "variant": variant,
                "gate": summary["gate"],
                "scenario_id": scenario["id"],
                "scenario_label": scenario["label"],
                "citation_key": scenario["citation_key"],
                "equivalent_t2_star_us": float(resolved["equivalent_t2_star_us"]),
                "tau_c_us": float(resolved["tau_c_us"]),
                "sigma_over_2pi_khz": float(resolved["sigma_over_2pi_khz"]),
                "average_gate_fidelity_mean": float(summary["average_gate_fidelity_mean"]),
                "average_gate_fidelity_stderr": float(summary["average_gate_fidelity_stderr"]),
                "survival_mean": float(summary["survival_mean"]),
                "quantile_05": float(quantiles[0]),
                "quantile_50": float(quantiles[1]),
                "quantile_95": float(quantiles[2]),
                "noiseless_average_gate_fidelity": float(
                    noiseless.get("average_gate_fidelity", np.nan)
                ),
                "dephasing_induced_average_fidelity_loss": float(
                    summary["dephasing_induced_average_fidelity_loss"]
                ),
                "n_realizations": int(summary["n_realizations"]),
                "source": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No experimental scenario summaries below {root}")
    return rows


def validate_pair(local_rows: list[dict[str, Any]], population_rows: list[dict[str, Any]]) -> None:
    local = {(r["gate"], r["t2_star_us"], r["tau_c_us"]) for r in local_rows}
    population = {(r["gate"], r["t2_star_us"], r["tau_c_us"]) for r in population_rows}
    if local != population:
        raise ValueError(
            f"Stage-one and stage-two grids differ: missing stage two={sorted(local-population)}, "
            f"missing stage one={sorted(population-local)}"
        )


def grouped_curves(rows: list[dict[str, Any]], gate: str) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["gate"] == gate:
            grouped[float(row["tau_c_us"])].append(row)
    return grouped


def plot_gate(
    population_rows: list[dict[str, Any]],
    population_scenarios: list[dict[str, Any]],
    *,
    gate: str,
    path: Path,
) -> None:
    """Original single-axis sweep style with experimental points added."""
    grouped = grouped_curves(population_rows, gate)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    curve_handles = []

    for tau in TAU_VALUES:
        if tau not in grouped:
            raise ValueError(f"Missing tau_c={tau:g} us for {gate}")
        points = sorted(grouped[tau], key=lambda row: row["t2_star_us"])
        x = np.asarray([row["t2_star_us"] for row in points], dtype=float)
        fidelity = np.asarray(
            [row["average_gate_fidelity_mean"] for row in points], dtype=float
        )
        sem = np.asarray(
            [row["average_gate_fidelity_stderr"] for row in points], dtype=float
        )
        q05 = np.asarray([row["quantile_05"] for row in points], dtype=float)
        q95 = np.asarray([row["quantile_95"] for row in points], dtype=float)

        infidelity = positive(1.0 - fidelity)
        lower = positive(1.0 - q95)
        upper = positive(1.0 - q05)
        line = ax.plot(
            x,
            infidelity,
            marker="o",
            label=rf"$\tau_c={tau:g}\,\mu\mathrm{{s}}$",
        )[0]
        ax.fill_between(x, lower, upper, alpha=0.12, color=line.get_color())
        ax.errorbar(
            x,
            infidelity,
            yerr=np.minimum(sem, 0.99 * infidelity),
            fmt="none",
            capsize=3,
            color=line.get_color(),
        )
        curve_handles.append(line)

    baseline_values = [
        1.0 - row["noiseless_average_gate_fidelity"]
        for row in population_rows
        if row["gate"] == gate
        and np.isfinite(row["noiseless_average_gate_fidelity"])
    ]
    if baseline_values:
        ax.axhline(
            float(positive(np.median(baseline_values))),
            linestyle=":",
            linewidth=1.2,
            label="Noiseless baseline",
        )

    scenario_handles = []
    for scenario_id in SCENARIO_ORDER:
        matches = [
            row
            for row in population_scenarios
            if row["gate"] == gate and row["scenario_id"] == scenario_id
        ]
        if not matches:
            continue
        row = matches[0]
        y = float(positive(1.0 - row["average_gate_fidelity_mean"]).item())
        handle = ax.errorbar(
            [row["equivalent_t2_star_us"]],
            [y],
            yerr=[min(row["average_gate_fidelity_stderr"], 0.99 * y)],
            marker=SCENARIO_MARKERS[scenario_id],
            markersize=8,
            linestyle="none",
            markeredgecolor="black",
            markeredgewidth=0.8,
            capsize=3,
            label=SCENARIO_SHORT[scenario_id],
            zorder=5,
        )
        scenario_handles.append(handle)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Electron Ramsey time $T_{2,A}^{\ast}$ ($\mu$s)")
    ax.set_ylabel(r"Average gate infidelity $1-\overline{F}_{\mathrm{avg}}$")
    ax.grid(True, which="both", alpha=0.25)

    curve_legend = ax.legend(
        handles=curve_handles,
        fontsize=8,
        loc="lower left",
        ncol=2,
        title=r"Correlation time $\tau_c$",
    )
    ax.add_artist(curve_legend)
    if scenario_handles:
        ax.legend(
            handles=scenario_handles,
            fontsize=8,
            loc="upper right",
            title="Experimental scenarios",
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_scenario_rows(
    local_rows: list[dict[str, Any]], population_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ref = {(row["gate"], row["scenario_id"]): row for row in local_rows}
    rob = {(row["gate"], row["scenario_id"]): row for row in population_rows}
    rows = []
    for scenario_id in SCENARIO_ORDER:
        for gate in GATES:
            r0 = ref[(gate, scenario_id)]
            r1 = rob[(gate, scenario_id)]
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_label": r1["scenario_label"],
                    "citation_key": r1["citation_key"],
                    "gate": gate,
                    "equivalent_t2_star_us": r1["equivalent_t2_star_us"],
                    "tau_c_us": r1["tau_c_us"],
                    "sigma_over_2pi_khz": r1["sigma_over_2pi_khz"],
                    "local_fidelity": r0["average_gate_fidelity_mean"],
                    "local_stderr": r0["average_gate_fidelity_stderr"],
                    "population_fidelity": r1["average_gate_fidelity_mean"],
                    "population_stderr": r1["average_gate_fidelity_stderr"],
                    "fidelity_improvement": (
                        r1["average_gate_fidelity_mean"]
                        - r0["average_gate_fidelity_mean"]
                    ),
                    "local_noise_loss": r0[
                        "dephasing_induced_average_fidelity_loss"
                    ],
                    "population_noise_loss": r1[
                        "dephasing_induced_average_fidelity_loss"
                    ],
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict[str, Any]], path: Path) -> None:
    by = {(row["scenario_id"], row["gate"]): row for row in rows}
    labels = {
        "bargill_12c_cvd": r"Bar-Gill purified $^{12}\mathrm C$",
        "hayashi_hpht_no8": r"Hayashi HPHT No.~8",
        "bauch_12c_2ppm": r"Bauch engineered $^{12}\mathrm C$",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Average gate fidelities in the three experimentally anchored material settings. Stage-one and population-refined pulses use identical noise seeds; uncertainties are one standard error.}",
        r"\label{tab:experimental_material_results}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Material & $(T_2^*,\tau_c)$ ($\mu$s) & $F_{ZZZ}^{\rm local}$ & $F_{ZZZ}^{\rm pop}$ & $F_{XZZ}^{\rm local}$ & $F_{XZZ}^{\rm pop}$ \\",
        r"\midrule",
    ]

    def cell(row: dict[str, Any], prefix: str) -> str:
        return (
            f"{row[f'{prefix}_fidelity']:.6f} "
            f"$\\pm$ {row[f'{prefix}_stderr']:.1e}"
        )

    for scenario_id in SCENARIO_ORDER:
        zzz = by[(scenario_id, "ZZZ")]
        xzz = by[(scenario_id, "XZZ")]
        pair = f"({zzz['equivalent_t2_star_us']:.3g},{zzz['tau_c_us']:.3g})"
        lines.append(
            f"{labels[scenario_id]} & ${pair}$ & "
            f"{cell(zzz, 'local')} & {cell(zzz, 'population')} & "
            f"{cell(xzz, 'local')} & {cell(xzz, 'population')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-grid-root", type=Path, required=True)
    parser.add_argument("--population-grid-root", type=Path, required=True)
    parser.add_argument("--local-scenario-root", type=Path, required=True)
    parser.add_argument("--population-scenario-root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("Bilder"))
    parser.add_argument("--table-dir", type=Path, default=Path("generated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    local_grid = load_grid_rows(resolve(project_root, args.local_grid_root))
    population_grid = load_grid_rows(resolve(project_root, args.population_grid_root))
    validate_pair(local_grid, population_grid)
    local_scenarios = load_scenario_rows(
        resolve(project_root, args.local_scenario_root), "local_equivalence"
    )
    population_scenarios = load_scenario_rows(
        resolve(project_root, args.population_scenario_root), "population_refined"
    )

    figure_dir = resolve(project_root, args.figure_dir)
    table_dir = resolve(project_root, args.table_dir)
    for gate in GATES:
        canonical = figure_dir / f"ou_fidelity_sweep_{gate}.png"
        plot_gate(
            population_grid,
            population_scenarios,
            gate=gate,
            path=canonical,
        )

    rows = paired_scenario_rows(local_scenarios, population_scenarios)
    write_csv(rows, table_dir / "experimental_scenario_results.csv")
    write_latex_table(rows, table_dir / "experimental_scenario_results.tex")
    (table_dir / "comparison_metadata.json").write_text(
        json.dumps(
            {
                "plotted_variant": "population_refined",
                "stage_one_usage": "matched experimental-scenario comparison table",
                "cartesian_grid": {
                    "t2_star_us": [2, 5, 10, 20, 50],
                    "tau_c_us": [0.1, 0.3, 1, 3, 15, 30],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Figures: {figure_dir}")
    print(f"Scenario table: {table_dir / 'experimental_scenario_results.tex'}")


if __name__ == "__main__":
    main()
