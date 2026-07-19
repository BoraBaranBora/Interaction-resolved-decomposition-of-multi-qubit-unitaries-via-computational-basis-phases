import math

import torch

from control_optimization.pulse import DirectFourierPulse, FourierPulseBounds, tukey_window


def make_model(basis_size: int = 4) -> DirectFourierPulse:
    return DirectFourierPulse(
        FourierPulseBounds(
            basis_size=basis_size,
            max_field_uT=210.0,
            min_angular_frequency=-2.0 * math.pi * 5.0e6,
            max_angular_frequency=2.0 * math.pi * 5.0e6,
            phase_bound=math.pi,
            taper_fraction=0.15,
        )
    )


def test_direct_drive_is_literal_manuscript_equation() -> None:
    model = make_model()
    time = torch.linspace(0.0, 1.5e-6, 401, dtype=torch.float64)
    physical = torch.cat(
        (
            torch.tensor([30.0, -22.0, 15.0, 8.0], dtype=torch.float64),
            2.0
            * math.pi
            * 1.0e6
            * torch.tensor([0.4, 1.1, 2.3, 4.2], dtype=torch.float64),
            torch.tensor([0.2, -0.5, 0.7, -1.2], dtype=torch.float64),
        )
    )
    raw = model.raw_from_physical(physical)
    amplitudes, frequencies, phases = model.unpack_raw(raw)
    expected = tukey_window(time, model.bounds.taper_fraction) * torch.sum(
        amplitudes[:, None]
        * torch.cos(frequencies[:, None] * time[None, :] + phases[:, None]),
        dim=0,
    )
    actual = model.drive(time, raw)
    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_direct_pulse_has_no_hidden_peak_rescaling() -> None:
    model = make_model(basis_size=3)
    time = torch.linspace(0.0, 1.0e-6, 301, dtype=torch.float64)
    physical = torch.cat(
        (
            torch.full((3,), 0.95 * model.bounds.max_field_uT, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
    )
    raw = model.raw_from_physical(physical)
    drive = model.drive(time, raw)
    assert drive.abs().max() > model.bounds.max_field_uT
    assert model.peak_penalty(drive) > 0


def test_direct_fit_recovers_synthetic_waveform() -> None:
    model = make_model(basis_size=4)
    time = torch.linspace(0.0, 1.5e-6, 501, dtype=torch.float64)
    physical = torch.cat(
        (
            torch.tensor([34.0, 21.0, 13.0, 7.0], dtype=torch.float64),
            2.0
            * math.pi
            * 1.0e6
            * torch.tensor([0.7, 1.5, 2.6, 4.0], dtype=torch.float64),
            torch.tensor([0.1, -0.4, 0.8, -1.0], dtype=torch.float64),
        )
    )
    target = model.drive(time, model.raw_from_physical(physical))
    fit = model.fit_to_waveform(
        time,
        target,
        steps=250,
        learning_rate=0.03,
        gradient_clip_norm=5.0,
        restarts=3,
        lbfgs_steps=30,
        seed=9,
    )
    assert fit.relative_error < 3.0e-2
    assert fit.peak_fraction <= 1.0 + 1.0e-6
