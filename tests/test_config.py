import json
from pathlib import Path

from control_optimization.config import load_control_config


def test_loads_warm_start_and_final_resolution(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            {
                "gate": "zzz",
                "duration_ns": 100.0,
                "basis_size": 3,
                "output_dir": "results/test",
                "final_steps_per_ns": 1.0,
                "warm_start": {"fit_steps": 123},
            }
        ),
        encoding="utf-8",
    )
    config = load_control_config(path)
    assert config.warm_start.fit_steps == 123
    assert config.final_steps_per_ns == 1.0
