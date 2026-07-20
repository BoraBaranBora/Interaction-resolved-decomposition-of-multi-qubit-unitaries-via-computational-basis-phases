from types import SimpleNamespace

import pytest
import torch

from control_optimization.config import ControlConfig, WarmStartConfig
from control_optimization.runner import ControlOptimizer


def minimal_config(**kwargs):
    values = dict(gate="zzz", duration_ns=10.0, basis_size=1, output_dir="out")
    values.update(kwargs)
    return ControlConfig(**values)


def test_warm_start_noise_must_be_nonnegative():
    with pytest.raises(ValueError, match="parameter_noise_std"):
        minimal_config(warm_start=WarmStartConfig(parameter_noise_std=-1.0)).validate()


def test_seeded_perturbation_is_reproducible():
    optimizer = object.__new__(ControlOptimizer)
    optimizer.config = SimpleNamespace(
        seed=17,
        warm_start=SimpleNamespace(parameter_noise_std=0.01),
    )
    optimizer.device = torch.device("cpu")
    optimizer.warm_start_info = {"source": "test"}
    raw = torch.zeros(7, dtype=torch.float64)
    first = optimizer._apply_seeded_parameter_perturbation(raw)
    optimizer.warm_start_info = {"source": "test"}
    second = optimizer._apply_seeded_parameter_perturbation(raw)
    assert torch.equal(first, second)
    assert not torch.equal(first, raw)
