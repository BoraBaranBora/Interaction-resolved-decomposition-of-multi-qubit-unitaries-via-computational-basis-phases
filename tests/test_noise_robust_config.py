import json
from pathlib import Path

from control_optimization.config import load_control_config


def test_population_100_weight_loads(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps({
            "gate": "zzz",
            "duration_ns": 100.0,
            "basis_size": 3,
            "output_dir": "results/test",
            "pulse_parameterization": "direct_fourier",
            "objective_weights": {"population_100_sum": 3.0e-4},
        }),
        encoding="utf-8",
    )
    config = load_control_config(path)
    assert config.objective_weights.population_100_sum == 3.0e-4
