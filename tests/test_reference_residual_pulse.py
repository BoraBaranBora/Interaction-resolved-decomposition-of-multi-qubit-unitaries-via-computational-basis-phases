import math

import torch

from control_optimization.pulse import (
    FourierPulseBounds,
    ReferenceResidualPulse,
)


def make_model() -> ReferenceResidualPulse:
    bounds = FourierPulseBounds(
        basis_size=5,
        max_field_uT=200.0,
        min_angular_frequency=-2.0 * math.pi * 5.0e6,
        max_angular_frequency=2.0 * math.pi * 5.0e6,
        phase_bound=math.pi,
        taper_fraction=0.15,
    )
    time = torch.linspace(0.0, 1.5e-6, 501, dtype=torch.float64)
    reference = 140.0 * torch.sin(2.0 * math.pi * 1.2e6 * time)
    return ReferenceResidualPulse(bounds, time, reference)


def test_zero_residual_reproduces_reference_on_original_grid() -> None:
    model = make_model()
    time = model.reference_time_grid.clone()
    raw = model.initial_raw(device="cpu")
    generated = model.drive(time, raw)
    assert torch.allclose(generated, model.reference_mw_drive, atol=1.0e-12, rtol=0.0)


def test_zero_residual_reproduces_interpolated_reference_on_new_grid() -> None:
    model = make_model()
    time = torch.linspace(0.0, 1.5e-6, 237, dtype=torch.float64)
    raw = model.initial_raw(device="cpu")
    generated = model.drive(time, raw)
    assert torch.allclose(generated, model.reference(time), atol=1.0e-12, rtol=0.0)


def test_residual_is_bounded_and_differentiable() -> None:
    model = make_model()
    time = torch.linspace(0.0, 1.5e-6, 301, dtype=torch.float64)
    raw = model.initial_raw(device="cpu").detach().clone()
    raw[: model.bounds.basis_size] = 0.2
    raw.requires_grad_(True)
    generated = model.drive(time, raw)
    assert generated.abs().max() <= model.bounds.max_field_uT + 1.0e-10
    loss = generated.square().mean()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.linalg.vector_norm(raw.grad[: model.bounds.basis_size]) > 0
