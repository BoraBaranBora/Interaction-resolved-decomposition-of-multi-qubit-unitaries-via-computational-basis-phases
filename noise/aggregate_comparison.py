"""Create paired reference/robust sweep figures with experimental overlays."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def validate_pair(reference: list[dict[str, Any]], robust: list[dict[str, Any]]) -> None:
    ref = {(r["gate"], r["t2_star_us"], r["tau_c_us"]) for r in reference}
    rob = {(r["gate"], r["t2_star_us"], r["tau_c_us"]) for r in robust}
    if ref != rob:
        raise ValueError(
            f"Reference and robust grids differ: missing robust={sorted(ref-rob)}, "
            f"missing reference={sorted(rob-ref)}"
        )


def curve_groups(rows: list[dict[str, Any]], gate: str) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["gate"] == gate:
            grouped[float(row["tau_c_us"])].append(row)
    return grouped


def plot_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    *,
    gate: str,
    title: str,
) -> None:
    grouped = curve_groups(rows, gate)
    line_handles = []
    for tau in TAU_VALUES:
        points = sorted(grouped[tau], key=lambda row: row["t2_star_us"])
        x = np.asarray([row["t2_star_us"] for row in points])
        y = positive(1.0 - np.asarray([row["average_gate_fidelity_mean"] for row in points]))
        sem = np.asarray([row["average_gate_fidelity_stderr"] for row in points])
        q_low = positive(1.0 - np.asarray([row["quantile_95"] for row in points]))
        q_high = positive(1.0 - np.asarray([row["quantile_05"] for row in points]))
        line = ax.plot(x, y, marker="o", linewidth=1.2, markersize=4,
                       label=rf"$\tau_c={tau:g}\,\mu\mathrm{{s}}$")[0]
        ax.fill_between(x, q_low, q_high, alpha=0.10, color=line.get_color())
        ax.errorbar(x, y, yerr=np.minimum(sem, 0.99 * y), fmt="none",
                    capsize=2, color=line.get_color(), linewidth=0.8)
        line_handles.append(line)

    baseline = np.nanmedian([
        1.0 - row["noiseless_average_gate_fidelity"] for row in rows
        if row["gate"] == gate
    ])
    if np.isfinite(baseline):
        ax.axhline(float(positive(baseline)), linestyle=":", linewidth=1.2)

    point_handles = []
    for scenario_id in SCENARIO_ORDER:
        matches = [
            row for row in scenarios
            if row["gate"] == gate and row["scenario_id"] == scenario_id
        ]
        if not matches:
            continue
        row = matches[0]
        x = row["equivalent_t2_star_us"]
        y = float(positive(1.0 - row["average_gate_fidelity_mean"]).item())
        yerr = row["average_gate_fidelity_stderr"]
        handle = ax.errorbar(
            [x], [y], yerr=[min(yerr, 0.99 * y)],
            marker=SCENARIO_MARKERS[scenario_id], markersize=8,
            linestyle="none", markeredgecolor="black", markeredgewidth=0.8,
            capsize=3, label=(
                f"{SCENARIO_SHORT[scenario_id]} "
                rf"($\tau_c={row['tau_c_us']:.3g}\,\mu\mathrm{{s}}$)"
            ), zorder=5,
        )
        point_handles.append(handle)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_title(title)
    ax.set_xlabel(r"Electron Ramsey time $T_{2,A}^{\ast}$ ($\mu$s)")
    ax.grid(True, which="both", alpha=0.25)
    tau_legend = ax.legend(handles=line_handles, fontsize=7, loc="lower left", ncol=2,
                           title="Cartesian sweep")
    ax.add_artist(tau_legend)
    if point_handles:
        ax.legend(handles=point_handles, fontsize=7, loc="upper right",
                  title="Experimental scenarios")


def plot_gate_comparison(
    reference_rows: list[dict[str, Any]],
    robust_rows: list[dict[str, Any]],
    reference_scenarios: list[dict[str, Any]],
    robust_scenarios: list[dict[str, Any]],
    *,
    gate: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharex=True, sharey=True)
    plot_panel(axes[0], reference_rows, reference_scenarios, gate=gate,
               title="Reference high-fidelity pulse")
    plot_panel(axes[1], robust_rows, robust_scenarios, gate=gate,
               title="Electron-exposure-penalized pulse")
    axes[0].set_ylabel(r"Average gate infidelity $1-\overline{F}_{\mathrm{avg}}$")
    fig.suptitle(f"{gate} Ramsey-calibrated OU sensitivity map")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_scenario_rows(
    reference: list[dict[str, Any]], robust: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ref = {(row["gate"], row["scenario_id"]): row for row in reference}
    rob = {(row["gate"], row["scenario_id"]): row for row in robust}
    rows = []
    for scenario_id in SCENARIO_ORDER:
        for gate in GATES:
            r0 = ref[(gate, scenario_id)]
            r1 = rob[(gate, scenario_id)]
            rows.append({
                "scenario_id": scenario_id,
                "scenario_label": r1["scenario_label"],
                "citation_key": r1["citation_key"],
                "gate": gate,
                "equivalent_t2_star_us": r1["equivalent_t2_star_us"],
                "tau_c_us": r1["tau_c_us"],
                "sigma_over_2pi_khz": r1["sigma_over_2pi_khz"],
                "reference_fidelity": r0["average_gate_fidelity_mean"],
                "reference_stderr": r0["average_gate_fidelity_stderr"],
                "robust_fidelity": r1["average_gate_fidelity_mean"],
                "robust_stderr": r1["average_gate_fidelity_stderr"],
                "fidelity_improvement": (
                    r1["average_gate_fidelity_mean"] - r0["average_gate_fidelity_mean"]
                ),
                "reference_noise_loss": r0["dephasing_induced_average_fidelity_loss"],
                "robust_noise_loss": r1["dephasing_induced_average_fidelity_loss"],
                "loss_reduction_fraction": 1.0 - (
                    r1["dephasing_induced_average_fidelity_loss"]
                    / max(r0["dephasing_induced_average_fidelity_loss"], 1e-15)
                ),
            })
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
        r"\caption{Gate performance in the three experimentally anchored material settings. Values are unconditional logical average fidelities; quoted uncertainties are one standard error. The reference and electron-exposure-penalized pulses are evaluated using identical noise seeds.}",
        r"\label{tab:experimental_material_results}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Material & $(T_2^*,\tau_c)$ ($\mu$s) & $F_{ZZZ}^{\rm ref}$ & $F_{ZZZ}^{\rm robust}$ & $F_{XZZ}^{\rm ref}$ & $F_{XZZ}^{\rm robust}$ \\",
        r"\midrule",
    ]
    for scenario_id in SCENARIO_ORDER:
        z = by[(scenario_id, "ZZZ")]
        x = by[(scenario_id, "XZZ")]
        pair = f"({z['equivalent_t2_star_us']:.3g},{z['tau_c_us']:.3g})"
        def cell(row: dict[str, Any], prefix: str) -> str:
            value = row[f"{prefix}_fidelity"]
            err = row[f"{prefix}_stderr"]
            return f"{value:.6f} $\\pm$ {err:.1e}"
        lines.append(
            f"{labels[scenario_id]} & ${pair}$ & {cell(z,'reference')} & {cell(z,'robust')} & {cell(x,'reference')} & {cell(x,'robust')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-grid-root", type=Path, required=True)
    parser.add_argument("--robust-grid-root", type=Path, required=True)
    parser.add_argument("--reference-scenario-root", type=Path, required=True)
    parser.add_argument("--robust-scenario-root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("Bilder"))
    parser.add_argument("--table-dir", type=Path, default=Path("results_ou"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    reference_grid = load_grid_rows(resolve(project_root, args.reference_grid_root))
    robust_grid = load_grid_rows(resolve(project_root, args.robust_grid_root))
    validate_pair(reference_grid, robust_grid)
    reference_scenarios = load_scenario_rows(
        resolve(project_root, args.reference_scenario_root), "reference"
    )
    robust_scenarios = load_scenario_rows(
        resolve(project_root, args.robust_scenario_root), "robust"
    )
    figure_dir = resolve(project_root, args.figure_dir)
    table_dir = resolve(project_root, args.table_dir)
    for gate in GATES:
        plot_gate_comparison(
            reference_grid, robust_grid,
            reference_scenarios, robust_scenarios,
            gate=gate,
            path=figure_dir / f"ou_fidelity_sweep_{gate}_reference_vs_robust.png",
        )
    rows = paired_scenario_rows(reference_scenarios, robust_scenarios)
    write_csv(rows, table_dir / "experimental_scenario_results.csv")
    write_latex_table(rows, table_dir / "experimental_scenario_results.tex")
    metadata = {
        "cartesian_grid": {
            "t2_star_us": [2, 5, 10, 20, 50],
            "tau_c_us": [0.1, 0.3, 1, 3, 15, 30],
        },
        "interpretation": (
            "Flattening toward the noiseless baseline at large T2* is the generic "
            "weak-noise limit. Control-induced robustness is established by the "
            "paired reference-versus-robust comparison at identical grid points."
        ),
        "n_table_rows": len(rows),
    }
    (table_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Figures: {figure_dir}")
    print(f"Scenario CSV: {table_dir / 'experimental_scenario_results.csv'}")
    print(f"LaTeX table: {table_dir / 'experimental_scenario_results.tex'}")


if __name__ == "__main__":
    main()
