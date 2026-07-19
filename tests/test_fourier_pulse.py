import math

import torch

from control_optimization.pulse import BoundedFourierPulse, FourierPulseBounds


def pulse() -> BoundedFourierPulse:
    return BoundedFourierPulse(
        FourierPulseBounds(
            basis_size=5,
            max_field_uT=200.0,
            min_angular_frequency=-2.0 * math.pi * 5.0e6,
            max_angular_frequency=2.0 * math.pi * 5.0e6,
            phase_bound=math.pi,
            taper_fraction=0.15,
        )
    )


def test_hard_bounds_and_zero_endpoints() -> None:
    model = pulse()
    raw = torch.linspace(-3.0, 3.0, model.bounds.n_parameters, dtype=torch.float64)
    time = torch.linspace(0.0, 1.5e-6, 501, dtype=torch.float64)
    drive = model.drive(time, raw)
    amplitudes, frequencies, phases = model.unpack_raw(raw)

    assert amplitudes.abs().max() <= model.bounds.component_amplitude_bound
    assert frequencies.min() >= model.bounds.min_angular_frequency
    assert frequencies.max() <= model.bounds.max_angular_frequency
    assert phases.abs().max() <= model.bounds.phase_bound
    assert abs(drive[0].item()) < 1.0e-15
    assert abs(drive[-1].item()) < 1.0e-15
    assert drive.abs().max() <= model.bounds.max_field_uT + 1.0e-15


def test_physical_raw_roundtrip() -> None:
    model = pulse()
    physical = torch.cat(
        (
            torch.linspace(-1.0, 1.0, 5) * 0.7 * model.bounds.component_amplitude_bound,
            torch.linspace(
                model.bounds.min_angular_frequency * 0.8,
                model.bounds.max_angular_frequency * 0.8,
                5,
            ),
            torch.linspace(-0.8 * math.pi, 0.8 * math.pi, 5),
        )
    ).to(torch.float64)
    raw = model.raw_from_physical(physical)
    reconstructed = model.physical_vector(raw)
    assert torch.allclose(physical, reconstructed, rtol=1.0e-10, atol=1.0e-10)


def test_drive_is_differentiable() -> None:
    model = pulse()
    raw = torch.zeros(model.bounds.n_parameters, dtype=torch.float64, requires_grad=True)
    time = torch.linspace(0.0, 1.0e-6, 101, dtype=torch.float64)
    loss = model.drive(time, raw).square().sum()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_saved_waveform_fit_recovers_synthetic_pulse() -> None:
    model = pulse()
    time = torch.linspace(0.0, 1.5e-6, 401, dtype=torch.float64)
    true_physical = torch.cat(
        (
            torch.tensor([0.25, -0.18, 0.12, 0.08, -0.05], dtype=torch.float64)
            * model.bounds.component_amplitude_bound,
            torch.tensor([0.5, 1.2, 2.0, 3.1, 4.2], dtype=torch.float64)
            * 2.0
            * math.pi
            * 1.0e6,
            torch.tensor([0.2, -0.5, 0.8, -1.0, 0.4], dtype=torch.float64),
        )
    )
    true_raw = model.raw_from_physical(true_physical)
    target = model.drive(time, true_raw)

    fit = model.fit_to_waveform(
        time,
        target,
        steps=400,
        learning_rate=0.04,
        gradient_clip_norm=10.0,
    )
    assert fit.relative_error < 2.0e-2
    assert fit.relative_error <= fit.initial_relative_error
    assert fit.peak_fraction <= 1.0 + 1.0e-12
