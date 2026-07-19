"""Source-native definitions of the three production material scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

from .ou import OUParameters


@dataclass(frozen=True)
class MaterialScenario:
    id: str
    label: str
    source: str
    citation_key: str
    material: str
    role: str
    calibration: dict[str, Any]
    notes: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaterialScenario":
        required = {
            "id", "label", "source", "citation_key", "material",
            "role", "calibration", "notes",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Scenario is missing: {', '.join(missing)}")
        if not isinstance(payload["calibration"], dict):
            raise TypeError("scenario calibration must be an object")
        return cls(**{key: payload[key] for key in required})

    def ou_parameters(self) -> OUParameters:
        kind = str(self.calibration.get("type", ""))
        if kind == "bar_gill_lorentzian":
            delta_khz = float(self.calibration["delta_khz"])
            tau_us = float(self.calibration["tau_c_us"])
            # Bar-Gill reports Delta as a frequency-width parameter in kHz.
            # We use the source's Lorentzian OU convention and identify it with
            # the rms longitudinal detuning width, then convert cycles/s to rad/s.
            return OUParameters.from_sigma(
                2.0 * math.pi * delta_khz * 1.0e3,
                tau_us * 1.0e-6,
                calibration="bar_gill_lorentzian_delta_as_rms_width",
            )

        if kind == "hayashi_lambda":
            lambda_mhz = float(self.calibration["lambda_mhz"])
            tau_us = float(self.calibration["tau_c_us"])
            # Hayashi uses H_I(t) = lambda f(t) sigma_z with unit-variance
            # exponentially correlated f(t).  Our convention is
            # H_noise(t) = beta(t) Z/2, hence beta(t) = 2 lambda f(t).
            # The plotted lambda is supplied in MHz (cycles/s), so the rms
            # angular-frequency width is sigma_beta = 4 pi lambda.
            return OUParameters.from_sigma(
                4.0 * math.pi * lambda_mhz * 1.0e6,
                tau_us * 1.0e-6,
                calibration="hayashi_same_sample_lambda_and_tau",
            )

        if kind == "bauch_concentration_scaling":
            nitrogen_ppm = float(self.calibration["nitrogen_ppm"])
            t2_star_scale = float(self.calibration["t2_star_us_ppm"])
            t2_scale = float(self.calibration["echo_t2_us_ppm"])
            t2_other = float(self.calibration["echo_t2_other_us"])
            t2_star_us = t2_star_scale / nitrogen_ppm
            echo_t2_us = 1.0 / (nitrogen_ppm / t2_scale + 1.0 / t2_other)
            return OUParameters.from_ramsey_and_echo(
                t2_star_us * 1.0e-6,
                echo_t2_us * 1.0e-6,
                calibration="bauch_measured_concentration_scaling_ou_inference",
            )

        raise ValueError(f"Unsupported scenario calibration type: {kind}")

    def resolved_metadata(self) -> dict[str, Any]:
        params = self.ou_parameters()
        metadata = {
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "citation_key": self.citation_key,
            "material": self.material,
            "role": self.role,
            "notes": self.notes,
            "source_calibration": self.calibration,
            "resolved_ou": {
                "sigma_rad_s": params.sigma_rad_s,
                "sigma_over_2pi_khz": params.sigma_rad_s / (2.0 * math.pi * 1.0e3),
                "tau_c_us": params.correlation_time_s * 1.0e6,
                "equivalent_t2_star_us": params.equivalent_t2_star_s * 1.0e6,
                "calibration": params.calibration,
            },
        }
        if self.calibration.get("type") == "bauch_concentration_scaling":
            n = float(self.calibration["nitrogen_ppm"])
            metadata["derived_characterization"] = {
                "nitrogen_ppm": n,
                "t2_star_us": float(self.calibration["t2_star_us_ppm"]) / n,
                "echo_t2_us": 1.0 / (
                    n / float(self.calibration["echo_t2_us_ppm"])
                    + 1.0 / float(self.calibration["echo_t2_other_us"])
                ),
            }
        return metadata


def load_scenarios(values: list[Any]) -> list[MaterialScenario]:
    if not isinstance(values, list) or not values:
        raise ValueError("noise_scenarios must be a nonempty list")
    scenarios = [MaterialScenario.from_dict(dict(value)) for value in values]
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("noise_scenario ids must be unique")
    # Resolve every calibration during config validation.
    for scenario in scenarios:
        scenario.ou_parameters()
    return scenarios
