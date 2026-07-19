"""Aggregate the absolute-time electron OU sweep and create manuscript figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


GATE_LABELS = ("ZZZ", "XZZ")


def resolve(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (project_root / value).resolve()


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("ensemble_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        model = summary.get("noise_model", {})
        quantiles = summary["average_gate_fidelity_quantiles_05_50_95"]
        noiseless = summary.get("noiseless", {})
        metadata = summary.get("regime_metadata", {})
        rows.append(
            {
                "gate": summary["gate"],
                "gate_key": summary.get("gate_key"),
                "pulse_checkpoint": summary.get("pulse_checkpoint"),
                "n_realizations": int(summary["n_realizations"]),
                "seed": int(summary["seed"]),
                "t2_star_us": round(float(model["t2_star_us"]), 10),
                "tau_c_us": round(float(model["correlation_time_us"]), 10),
                "sigma_rad_s": float(model["sigma_rad_s"]),
                "t2_status": metadata.get("t2_star", {}).get("status", "unspecified"),
                "t2_citation_key": metadata.get("t2_star", {}).get("citation_key"),
                "tau_status": metadata.get("tau_c", {}).get("status", "unspecified"),
                "tau_citation_key": metadata.get("tau_c", {}).get("citation_key"),
                "noiseless_entanglement_fidelity": float(
                    noiseless.get("entanglement_fidelity", np.nan)
                ),
                "noiseless_average_gate_fidelity": float(
                    noiseless.get("average_gate_fidelity", np.nan)
                ),
                "entanglement_fidelity_mean": float(
                    summary["entanglement_fidelity_mean"]
                ),
                "entanglement_fidelity_stderr": float(
                    summary["entanglement_fidelity_stderr"]
                ),
                "average_gate_fidelity_mean": float(
                    summary["average_gate_fidelity_mean"]
                ),
                "average_gate_fidelity_stderr": float(
                    summary["average_gate_fidelity_stderr"]
                ),
                "survival_mean": float(summary["survival_mean"]),
                "survival_stderr": float(summary["survival_stderr"]),
                "quantile_05": float(quantiles[0]),
                "quantile_50": float(quantiles[1]),
                "quantile_95": float(quantiles[2]),
                "dephasing_induced_average_fidelity_loss": float(
                    summary["dephasing_induced_average_fidelity_loss"]
                ),
                "source": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No ensemble_summary.json files below {root}")
    return rows


def expected_grid(root: Path) -> set[tuple[str, float, float]] | None:
    path = root / "sweep_manifest.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))["config"]
    result: set[tuple[str, float, float]] = set()
    labels = {"diagonal": "ZZZ", "nondiagonal": "XZZ"}
    for gate in config["gates"]:
        for t2 in config["t2_star_regimes"]:
            for tau in config["tau_c_regimes"]:
                result.add(
                    (
                        labels[gate],
                        round(float(t2["value_us"]), 10),
                        round(float(tau["value_us"]), 10),
                    )
                )
    return result


def validate_grid(
    rows: list[dict[str, Any]],
    expected: set[tuple[str, float, float]] | None,
    *,
    allow_incomplete: bool,
) -> None:
    observed: dict[tuple[str, float, float], int] = defaultdict(int)
    for row in rows:
        observed[(row["gate"], row["t2_star_us"], row["tau_c_us"])] += 1
    duplicates = [key for key, count in observed.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sweep points: {duplicates}")
    if expected is None:
        return
    missing = sorted(expected.difference(observed))
    extra = sorted(set(observed).difference(expected))
    if extra:
        raise ValueError(f"Unexpected sweep points: {extra}")
    if missing and not allow_incomplete:
        raise ValueError(f"Incomplete sweep; missing points: {missing}")
    if missing:
        print(f"Warning: {len(missing)} sweep points are missing")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tau_label(tau: float, status: str) -> str:
    suffixes = {
        "literature_modeled": "modeled",
        "experimentally_observed_bath": "observed bath",
        "fast_sensitivity_bound": "fast bound",
        "fast_correlated_sensitivity": "fast",
        "gate_timescale_crossover": "gate-scale",
        "moderately_slow_sensitivity": "moderately slow",
        "slow_sensitivity": "slow",
    }
    suffix = suffixes.get(status, status.replace("_", " "))
    return rf"$\tau_c={tau:g}\,\mu\mathrm{{s}}$ ({suffix})"


def positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, np.finfo(float).tiny)


def plot_gate(rows: list[dict[str, Any]], gate: str, path: Path) -> None:
    selected = [row for row in rows if row["gate"] == gate]
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[row["tau_c_us"]].append(row)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tau in sorted(grouped):
        points = sorted(grouped[tau], key=lambda item: item["t2_star_us"])
        x = np.asarray([item["t2_star_us"] for item in points], dtype=float)
        mean_fidelity = np.asarray(
            [item["average_gate_fidelity_mean"] for item in points], dtype=float
        )
        sem = np.asarray(
            [item["average_gate_fidelity_stderr"] for item in points], dtype=float
        )
        q05 = np.asarray([item["quantile_05"] for item in points], dtype=float)
        q95 = np.asarray([item["quantile_95"] for item in points], dtype=float)
        mean_infidelity = positive(1.0 - mean_fidelity)
        lower = positive(1.0 - q95)
        upper = positive(1.0 - q05)
        status = str(points[0]["tau_status"])
        line_style = "-" if status in {
            "literature_modeled", "experimentally_observed_bath"
        } else "--"
        line = ax.plot(
            x,
            mean_infidelity,
            marker="o",
            linestyle=line_style,
            label=tau_label(tau, status),
        )[0]
        ax.fill_between(x, lower, upper, alpha=0.12, color=line.get_color())
        ax.errorbar(
            x,
            mean_infidelity,
            yerr=np.minimum(sem, 0.99 * mean_infidelity),
            fmt="none",
            capsize=3,
            color=line.get_color(),
        )

    baseline_values = [
        1.0 - row["noiseless_average_gate_fidelity"]
        for row in selected
        if np.isfinite(row["noiseless_average_gate_fidelity"])
    ]
    if baseline_values:
        baseline = float(np.median(baseline_values))
        ax.axhline(
            positive(np.asarray([baseline]))[0],
            linestyle=":",
            linewidth=1.2,
            label="Noiseless baseline",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Electron Ramsey time $T_{2,A}^{\ast}$ ($\mu$s)")
    ax.set_ylabel(r"Average gate infidelity $1-\overline{F}_{\mathrm{avg}}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gate_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    """Compare dephasing-induced fidelity loss for both gates."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    marker = {"ZZZ": "o", "XZZ": "s"}
    for gate in GATE_LABELS:
        gate_rows = [row for row in rows if row["gate"] == gate]
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in gate_rows:
            grouped[row["tau_c_us"]].append(row)
        for tau in sorted(grouped):
            points = sorted(grouped[tau], key=lambda item: item["t2_star_us"])
            x = np.asarray([item["t2_star_us"] for item in points], dtype=float)
            loss = np.asarray(
                [item["dephasing_induced_average_fidelity_loss"] for item in points],
                dtype=float,
            )
            ax.plot(
                x,
                positive(loss),
                marker=marker[gate],
                linestyle="-" if gate == "ZZZ" else "--",
                label=rf"{gate}, $\tau_c={tau:g}\,\mu\mathrm{{s}}$",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Electron Ramsey time $T_{2,A}^{\ast}$ ($\mu$s)")
    ax.set_ylabel(r"Dephasing-induced loss $F_{\mathrm{avg},0}-\overline{F}_{\mathrm{avg}}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def reference_condition_table(rows: list[dict[str, Any]], path: Path) -> None:
    reference = [
        row for row in rows
        if row["t2_star_us"] == 5.0 and row["tau_c_us"] == 15.0
    ]
    if not reference:
        return
    fields = [
        "gate",
        "n_realizations",
        "entanglement_fidelity_mean",
        "average_gate_fidelity_mean",
        "survival_mean",
        "average_gate_fidelity_stderr",
        "quantile_05",
        "quantile_50",
        "quantile_95",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(reference, key=lambda item: item["gate"]):
            writer.writerow({field: row[field] for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("Bilder"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = resolve(project_root, args.root)
    figure_dir = resolve(project_root, args.figure_dir)
    rows = load_rows(root)
    rows.sort(key=lambda row: (row["gate"], row["tau_c_us"], row["t2_star_us"]))
    validate_grid(rows, expected_grid(root), allow_incomplete=args.allow_incomplete)

    write_csv(rows, root / "sweep_summary.csv")
    (root / "sweep_summary.json").write_text(
        json.dumps(
            {
                "n_points": len(rows),
                "gates": sorted({row["gate"] for row in rows}),
                "n_realizations": sorted({row["n_realizations"] for row in rows}),
                "t2_star_us": sorted({row["t2_star_us"] for row in rows}),
                "tau_c_us": sorted({row["tau_c_us"] for row in rows}),
                "noise_model": "electron_longitudinal_ornstein_uhlenbeck",
                "calibration": "ramsey_t2_star",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    reference_condition_table(rows, root / "reference_condition_T2star5_tauc15.csv")
    plot_gate(rows, "ZZZ", figure_dir / "ou_fidelity_sweep_ZZZ.png")
    plot_gate(rows, "XZZ", figure_dir / "ou_fidelity_sweep_XZZ.png")
    plot_gate_comparison(rows, figure_dir / "ou_dephasing_loss_comparison.png")

    print(f"Aggregated {len(rows)} points")
    print(f"CSV: {root / 'sweep_summary.csv'}")
    print(f"Reference table: {root / 'reference_condition_T2star5_tauc15.csv'}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
