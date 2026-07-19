import math
from pathlib import Path

import torch

from control_optimization.pulse import DirectFourierPulse, FourierPulseBounds
from evaluation.verify_fourier import verify


def test_verifier_accepts_exact_direct_checkpoint(tmp_path: Path) -> None:
    bounds = FourierPulseBounds(
        basis_size=3,
        max_field_uT=200.0,
        min_angular_frequency=-2.0 * math.pi * 5.0e6,
        max_angular_frequency=2.0 * math.pi * 5.0e6,
        phase_bound=math.pi,
        taper_fraction=0.15,
    )
    model = DirectFourierPulse(bounds)
    physical = torch.cat(
        (
            torch.tensor([30.0, 20.0, 10.0], dtype=torch.float64),
            2.0
            * math.pi
            * 1.0e6
            * torch.tensor([0.8, 2.0, 4.0], dtype=torch.float64),
            torch.tensor([0.2, -0.5, 0.7], dtype=torch.float64),
        )
    )
    raw = model.raw_from_physical(physical)
    time = torch.linspace(0.0, 1.0e-6, 301, dtype=torch.float64)
    drive = model.drive(time, raw)
    checkpoint = {
        "pulse_parameterization": "direct_fourier",
        "params": model.physical_vector(raw),
        "time_grid": time,
        "drive": [drive, torch.zeros_like(drive)],
        "pulse_settings": [
            {
                "basis_size": 3,
                "maximal_pulse": bounds.max_field_uT,
                "minimal_frequency": bounds.min_angular_frequency,
                "maximal_frequency": bounds.max_angular_frequency,
                "maximal_phase": bounds.phase_bound,
                "taper_fraction": bounds.taper_fraction,
            }
        ],
    }
    torch.save(checkpoint, tmp_path / "pulse_solution.pt")
    report = verify(tmp_path, tolerance=1.0e-10, peak_tolerance=1.0e-6)
    assert report["equation_exact"]
    assert report["peak_constraint_satisfied"]
