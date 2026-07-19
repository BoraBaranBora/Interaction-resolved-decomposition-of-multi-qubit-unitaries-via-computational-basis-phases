from __future__ import annotations

import math

import numpy as np
import torch

from noise.ou import OUParameters, sample_ou_process


def test_ramsey_calibration_reaches_exp_minus_one() -> None:
    parameters = OUParameters.from_t2_star(5e-6, 15e-6)
    observed = float(parameters.ramsey_coherence(5e-6))
    assert math.isclose(observed, math.exp(-1.0), rel_tol=1e-12, abs_tol=1e-12)


def test_stationary_variance_and_reproducibility() -> None:
    parameters = OUParameters.from_t2_star(5e-6, 15e-6)
    time = torch.linspace(0.0, 100e-6, 401, dtype=torch.float64)
    first = sample_ou_process(time, parameters, n_realizations=4000, seed=17)
    second = sample_ou_process(time, parameters, n_realizations=4000, seed=17)
    assert torch.equal(first, second)
    variance = float(first[:, -1].var(unbiased=True))
    assert np.isclose(variance, parameters.sigma_rad_s**2, rtol=0.06)


def test_nonuniform_time_grid_supported() -> None:
    parameters = OUParameters.from_t2_star(10e-6, 30e-6)
    time = torch.tensor([0.0, 1e-9, 3e-9, 8e-9, 20e-9], dtype=torch.float64)
    trace = sample_ou_process(time, parameters, n_realizations=3, seed=4)
    assert trace.shape == (3, 5)
    assert torch.isfinite(trace).all()
