import torch

from control_optimization.pulse import BoundedFourierPulse, FourierPulseBounds


def test_waveform_fit_accepts_jittered_grid():
    bounds = FourierPulseBounds(
        basis_size=3,
        max_field_uT=1.0,
        min_angular_frequency=-2.0e7,
        max_angular_frequency=2.0e7,
        phase_bound=3.141592653589793,
        taper_fraction=0.15,
    )
    pulse = BoundedFourierPulse(bounds)
    base = torch.linspace(0.0, 1.0e-6, 401, dtype=torch.float64)
    jitter = torch.zeros_like(base)
    jitter[1:-1] = 2.0e-12 * torch.sin(torch.arange(399, dtype=torch.float64))
    grid = torch.sort(base + jitter).values
    physical = torch.tensor(
        [
            0.32,
            -0.21,
            0.15,
            2.0e6,
            7.0e6,
            1.3e7,
            0.2,
            -0.7,
            1.1,
        ],
        dtype=torch.float64,
    )
    target = pulse.drive(grid, pulse.raw_from_physical(physical))
    fit = pulse.fit_to_waveform(
        grid,
        target,
        steps=250,
        learning_rate=0.03,
        gradient_clip_norm=10.0,
        initial_raw=pulse.raw_from_physical(physical),
    )
    assert fit.relative_error < 1.0e-6
