from __future__ import annotations

from pathlib import Path

from noise.sweep import load_config


def test_production_grid_uses_absolute_correlation_times() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "ou_electron_ramsey_grid.json")
    assert [entry["value_us"] for entry in config["t2_star_regimes"]] == [
        2.0, 5.0, 10.0, 20.0, 50.0
    ]
    assert [entry["value_us"] for entry in config["tau_c_regimes"]] == [
        0.1, 0.3, 1.0, 3.0, 15.0, 30.0
    ]
    assert config["gates"]["diagonal"]["pulse_dir"] == "results/control_zzz_direct"
    assert config["n_realizations"] == 256
    assert config["propagation_steps_per_ns"] == 1.0


def test_primary_config_contains_no_echo_or_nuclear_channels() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "ou_electron_ramsey_grid.json").read_text(
        encoding="utf-8"
    ).lower()
    assert "echo" not in text
    assert "c1:" not in text
    assert "c2:" not in text
