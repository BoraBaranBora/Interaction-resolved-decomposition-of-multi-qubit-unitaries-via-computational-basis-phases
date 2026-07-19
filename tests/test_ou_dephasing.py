import math
import numpy as np
import torch

from ou_dephasing import OUParameters, sample_ou_process


def test_ramsey_calibration_hits_exp_minus_one():
    params = OUParameters.from_t2_star(12e-6, 30e-6)
    value = float(params.ramsey_coherence(12e-6))
    assert abs(value - math.exp(-1.0)) < 1e-12


def test_echo_calibration_hits_exp_minus_one():
    params = OUParameters.from_hahn_echo_t2(300e-6, 100e-6)
    value = float(params.hahn_echo_coherence(300e-6))
    assert abs(value - math.exp(-1.0)) < 1e-12


def test_exact_update_preserves_stationary_variance():
    params = OUParameters(correlation_time_s=2e-6, sigma_rad_s=2.5e5)
    times = torch.linspace(0.0, 20e-6, 401)
    traces = sample_ou_process(times, params, n_realizations=20000, seed=123)
    final_std = float(traces[:, -1].std(unbiased=True))
    assert abs(final_std / params.sigma_rad_s - 1.0) < 0.025


def test_monte_carlo_ramsey_matches_analytic():
    params = OUParameters.from_t2_star(8e-6, 20e-6)
    times = torch.linspace(0.0, 8e-6, 161)
    traces = sample_ou_process(times, params, n_realizations=30000, seed=321)
    dt = torch.diff(times)
    phase = torch.sum(0.5 * (traces[:, 1:] + traces[:, :-1]) * dt, dim=1)
    coherence_mc = torch.exp(-1j * phase).mean().abs().item()
    coherence_exact = float(params.ramsey_coherence(8e-6))
    assert abs(coherence_mc - coherence_exact) < 0.02
