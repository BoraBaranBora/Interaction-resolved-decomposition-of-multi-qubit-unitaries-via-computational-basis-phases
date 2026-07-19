import json
from pathlib import Path

from control_optimization.config import load_control_config


def test_loads_direct_fourier_settings(tmp_path: Path) -> None:
    path = tmp_path / "direct.json"
    path.write_text(
        json.dumps(
            {
                "gate": "zzz",
                "duration_ns": 100.0,
                "basis_size": 11,
                "output_dir": "results/direct",
                "pulse_parameterization": "direct_fourier",
                "objective_weights": {"peak": 100.0},
                "warm_start": {
                    "fit_restarts": 4,
                    "fit_lbfgs_steps": 20,
                    "accept_imperfect_fit": True,
                    "minimum_corrected_fidelity": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_control_config(path)
    assert config.pulse_parameterization == "direct_fourier"
    assert config.objective_weights.peak == 100.0
    assert config.warm_start.fit_restarts == 4
    assert config.warm_start.fit_lbfgs_steps == 20
    assert config.warm_start.accept_imperfect_fit
