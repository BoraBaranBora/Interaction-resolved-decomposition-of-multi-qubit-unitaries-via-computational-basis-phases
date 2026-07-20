from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ControlConfig
from .objective import ObjectiveResult, SupportSelectiveObjective
from .pulse import (
    BoundedFourierPulse,
    DirectFourierPulse,
    FourierPulseBounds,
    ReferenceResidualPulse,
)
from .trajectory import TrajectoryMetrics, propagate_with_population_100_sum


def make_three_qubit_basis_indices(
    pc: dict[str, Any],
    *,
    carbon_pair: tuple[int, int],
    mI_block: int,
    electron_map: tuple[str, str],
) -> list[int]:
    active = list(pc["c_indices"])
    n_carbon = int(pc["N_C"])
    nconf = 2**n_carbon
    dim_nuc = 3 * nconf

    for carbon in carbon_pair:
        if carbon not in active:
            raise ValueError(f"Carbon {carbon} is not in active set {active}.")
    pos_b = active.index(carbon_pair[0])
    pos_c = active.index(carbon_pair[1])

    def carbon_bits(b: int, c: int) -> int:
        return (b << pos_b) | (c << pos_c)

    def electron_offset(a: int) -> int:
        if electron_map == ("m1", "0"):
            return dim_nuc if a == 0 else 0
        if electron_map == ("0", "m1"):
            return 0 if a == 0 else dim_nuc
        raise ValueError("Unsupported electron_map.")

    def index(a: int, b: int, c: int) -> int:
        return electron_offset(a) + mI_block * nconf + carbon_bits(b, c)

    return [
        index(0, 0, 0),
        index(0, 0, 1),
        index(0, 1, 0),
        index(0, 1, 1),
        index(1, 0, 0),
        index(1, 0, 1),
        index(1, 1, 0),
        index(1, 1, 1),
    ]


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ControlOptimizer:
    def __init__(self, config: ControlConfig, project_root: Path):
        self.config = config
        self.project_root = project_root.resolve()

        src = self.project_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        import quantum_model_NV as nv
        from evolution import get_propagator, get_time_grid

        self.nv = nv
        self.get_propagator = get_propagator
        self.get_time_grid = get_time_grid

        nv.set_active_carbons(list(config.active_carbons))
        self.pc = nv.get_precomp()
        self.device = torch.device(nv.device)
        self.complex_dtype = nv.dtype
        self.real_dtype = torch.float64
        self.delta_e = float(nv.detuning_for_target_all_up())
        self.omega_rf = torch.as_tensor(nv.ω1, device=self.device)
        dim_nuc = 3 * (2 ** int(self.pc["N_C"]))
        self.full_dimension = 2 * dim_nuc

        self.basis_indices = make_three_qubit_basis_indices(
            self.pc,
            carbon_pair=config.logical_carbons,
            mI_block=config.mI_block,
            electron_map=config.electron_map,
        )

        gamma_e = float(torch.as_tensor(nv.γ_e).detach().cpu())
        # In the public NV model, gamma_e is numerically in rad s^-1 uT^-1,
        # so the control arrays and saved waveforms are in microtesla.
        max_field_uT = 2.0 * math.pi * config.max_rabi_mhz * 1.0e6 / gamma_e
        bounds = FourierPulseBounds(
            basis_size=config.basis_size,
            max_field_uT=max_field_uT,
            min_angular_frequency=2.0 * math.pi * config.min_frequency_mhz * 1.0e6,
            max_angular_frequency=2.0 * math.pi * config.max_frequency_mhz * 1.0e6,
            phase_bound=config.phase_bound_pi * math.pi,
            taper_fraction=config.taper_fraction,
        )
        if config.pulse_parameterization.lower() == "direct_fourier":
            self.pulse = DirectFourierPulse(bounds)
        else:
            self.pulse = BoundedFourierPulse(bounds)
        self.objective = SupportSelectiveObjective(
            basis_indices=self.basis_indices,
            gate=config.gate,
            target_angle=config.target_angle_rad,
            pair_weight=config.pair_weight,
            tripartite_weight=config.tripartite_weight,
            weights=config.objective_weights,
            dtype=self.complex_dtype,
            device=self.device,
        )

        self.history: list[dict[str, Any]] = []
        self._evaluation_counter = 0
        self.warm_start_info: dict[str, Any] = {"source": "seeded_random"}

    def _time_grid(self, steps_per_ns: float) -> torch.Tensor:
        grid = self.get_time_grid(self.config.duration_ns, steps_per_ns)
        return grid.to(dtype=self.real_dtype, device=self.device)

    def _get_u(self, controls: list[torch.Tensor], dt: float, t: float) -> torch.Tensor:
        return self.nv.get_U_RWA(
            controls,
            dt,
            t,
            Δ_e=self.delta_e,
            ω_RF=self.omega_rf,
        )

    def _trajectory(
        self,
        controls: list[torch.Tensor],
        time_grid: torch.Tensor,
        *,
        return_traces: bool = False,
    ) -> TrajectoryMetrics:
        return propagate_with_population_100_sum(
            self._get_u,
            time_grid,
            controls,
            basis_indices=self.basis_indices,
            dimension=self.full_dimension,
            return_traces=return_traces,
        )

    def evaluate_controls(
        self, controls: list[torch.Tensor], time_grid: torch.Tensor
    ) -> tuple[ObjectiveResult, torch.Tensor]:
        fluence, smoothness = self.pulse.regularization(controls[0])
        trajectory = self._trajectory(controls, time_grid)
        result = self.objective(
            trajectory.propagator,
            fluence=fluence,
            smoothness=smoothness,
            peak_penalty=self.pulse.peak_penalty(controls[0]),
            population_100_sum=trajectory.population_100_sum,
        )
        return result, trajectory.propagator

    def evaluate(self, raw: torch.Tensor, time_grid: torch.Tensor) -> tuple[ObjectiveResult, list[torch.Tensor], torch.Tensor]:
        controls = self.pulse.controls(time_grid, raw)
        result, propagator = self.evaluate_controls(controls, time_grid)
        return result, controls, propagator

    def _record(self, stage: str, result: ObjectiveResult, raw: torch.Tensor) -> None:
        loss_value = float(result.loss.detach().cpu())
        row: dict[str, Any] = {
            "stage": stage,
            "evaluation": self._evaluation_counter,
            "loss": loss_value,
        }
        row.update(
            {
                key: float(value.detach().cpu())
                for key, value in result.components.items()
            }
        )
        row.update(
            {
                f"coordinate_{key}": float(value.detach().cpu())
                for key, value in result.coordinates.items()
            }
        )
        self.history.append(row)
        self._evaluation_counter += 1


    def _parameterization_name(self) -> str:
        if isinstance(self.pulse, ReferenceResidualPulse):
            return "reference_residual_fourier"
        if isinstance(self.pulse, DirectFourierPulse):
            return "direct_fourier"
        return "bounded_fourier"

    @staticmethod
    def _waveform_fingerprint(time_grid: torch.Tensor, mw_drive: torch.Tensor) -> str:
        digest = hashlib.sha256()
        for tensor in (time_grid, mw_drive):
            contiguous = tensor.detach().to(dtype=torch.float64, device="cpu").contiguous()
            digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()

    def _warm_start_cache_path(self) -> Path:
        configured = Path(self.config.output_dir)
        output_dir = configured if configured.is_absolute() else self.project_root / configured
        return output_dir / "direct_fourier_warm_start.pt"

    def _load_initial_raw(self) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed)

        def random_initialization(source: str = "seeded_random") -> torch.Tensor:
            self.warm_start_info = {"source": source}
            return self.pulse.initial_raw(
                device=self.device,
                dtype=self.real_dtype,
                generator=generator,
            )

        if not self.config.resume_from:
            return random_initialization()

        requested = self.config.pulse_parameterization.lower()
        result_dir = self.project_root / self.config.resume_from
        checkpoint_path = result_dir / "pulse_solution.pt"
        checkpoint: dict[str, Any] | None = None
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            except Exception as exc:
                print(f"[warning] Could not load {checkpoint_path}: {exc}")

        checkpoint_parameterization = (
            str(checkpoint.get("pulse_parameterization", "")).lower()
            if checkpoint is not None
            else ""
        )

        # Exact resume is valid only when the checkpoint and requested pulse
        # parameterizations agree.  A residual raw vector must never be treated
        # as a direct Fourier coefficient vector merely because both have 3N
        # entries.
        if checkpoint is not None and checkpoint.get("params_raw") is not None:
            candidate = torch.as_tensor(
                checkpoint["params_raw"], dtype=self.real_dtype
            ).reshape(-1)
            if (
                candidate.numel() == self.pulse.bounds.n_parameters
                and checkpoint_parameterization == requested
            ):
                if requested == "reference_residual_fourier":
                    if (
                        checkpoint.get("reference_time_grid") is None
                        or checkpoint.get("reference_mw_drive") is None
                    ):
                        raise ValueError(
                            "Residual checkpoint is missing its immutable reference waveform."
                        )
                    self.pulse = ReferenceResidualPulse(
                        self.pulse.bounds,
                        checkpoint["reference_time_grid"],
                        checkpoint["reference_mw_drive"],
                    )
                self.warm_start_info = {
                    "source": "exact_params_raw_resume",
                    "checkpoint": str(checkpoint_path),
                    "parameterization": requested,
                }
                print(f"Selected warm start: exact {requested} checkpoint parameters")
                return candidate.to(self.device)

        old_grid = old_mw = old_rf = None
        reference_result = None
        if (
            checkpoint is not None
            and checkpoint.get("drive") is not None
            and checkpoint.get("time_grid") is not None
        ):
            try:
                old_grid = torch.as_tensor(
                    checkpoint["time_grid"], dtype=self.real_dtype, device=self.device
                ).reshape(-1)
                old_drive = checkpoint["drive"]
                old_mw = torch.as_tensor(
                    old_drive[0] if isinstance(old_drive, (list, tuple)) else old_drive,
                    dtype=self.real_dtype,
                    device=self.device,
                ).reshape(-1)
                if isinstance(old_drive, (list, tuple)) and len(old_drive) > 1:
                    old_rf = torch.as_tensor(
                        old_drive[1], dtype=self.real_dtype, device=self.device
                    ).reshape(-1)
                else:
                    old_rf = torch.zeros_like(old_mw)
                if old_grid.shape != old_mw.shape or old_rf.shape != old_mw.shape:
                    raise ValueError(
                        "Saved time grid and control channels have inconsistent shapes."
                    )
                if self.config.warm_start.evaluate_reference:
                    with torch.no_grad():
                        reference_result, _ = self.evaluate_controls(
                            [old_mw, old_rf], old_grid
                        )
                    print(
                        "Reference waveform under new objective: "
                        f"loss={float(reference_result.loss.detach().cpu()):.6e}, "
                        f"F_corr={float(reference_result.components['corrected_fidelity'].detach().cpu()):.10f}, "
                        f"phi_ABC={float(reference_result.coordinates['ABC'].detach().cpu()):+.6f}"
                    )
            except Exception as exc:
                print(f"[warning] Could not evaluate the saved waveform: {exc}")
                old_grid = old_mw = old_rf = None

        if (
            requested == "reference_residual_fourier"
            and self.config.warm_start.enabled
            and old_grid is not None
            and old_mw is not None
        ):
            self.pulse = ReferenceResidualPulse(
                self.pulse.bounds, old_grid.detach().cpu(), old_mw.detach().cpu()
            )
            raw = self.pulse.initial_raw(
                device=self.device, dtype=self.real_dtype, generator=generator
            )
            with torch.no_grad():
                reconstructed = self.pulse.drive(old_grid, raw)
                relative_error = torch.linalg.vector_norm(
                    reconstructed - old_mw
                ) / torch.clamp(torch.linalg.vector_norm(old_mw), min=1.0e-30)
            self.warm_start_info = {
                "source": "reference_waveform_plus_fourier_residual",
                "checkpoint": str(checkpoint_path),
                "relative_waveform_error": float(relative_error.cpu()),
            }
            print(
                "Selected warm start: exact reference waveform + bounded Fourier residual "
                f"(waveform_error={float(relative_error.cpu()):.3e})"
            )
            return raw

        if (
            requested == "direct_fourier"
            and self.config.warm_start.enabled
            and old_grid is not None
            and old_mw is not None
        ):
            fingerprint = self._waveform_fingerprint(old_grid, old_mw)
            cache_path = self._warm_start_cache_path()
            if self.config.warm_start.cache_fit and cache_path.exists():
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    cached_raw = torch.as_tensor(
                        cached["params_raw"], dtype=self.real_dtype
                    ).reshape(-1)
                    expected_bounds = {
                        "max_field_uT": self.pulse.bounds.max_field_uT,
                        "min_angular_frequency": self.pulse.bounds.min_angular_frequency,
                        "max_angular_frequency": self.pulse.bounds.max_angular_frequency,
                        "phase_bound": self.pulse.bounds.phase_bound,
                        "taper_fraction": self.pulse.bounds.taper_fraction,
                    }
                    cached_bounds = cached.get("bounds", {})
                    bounds_match = all(
                        math.isclose(
                            float(cached_bounds.get(key, math.nan)),
                            float(value),
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                        for key, value in expected_bounds.items()
                    )
                    if (
                        cached.get("waveform_fingerprint") == fingerprint
                        and int(cached.get("basis_size", -1)) == self.config.basis_size
                        and cached_raw.numel() == self.pulse.bounds.n_parameters
                        and bounds_match
                    ):
                        cached_raw = cached_raw.to(self.device)
                        with torch.no_grad():
                            cached_result, _, _ = self.evaluate(cached_raw, old_grid)
                            reconstructed = self.pulse.drive(old_grid, cached_raw)
                            error = torch.linalg.vector_norm(
                                reconstructed - old_mw
                            ) / torch.clamp(
                                torch.linalg.vector_norm(old_mw), min=1.0e-30
                            )
                        self.warm_start_info = {
                            "source": "cached_direct_fourier_fit",
                            "checkpoint": str(checkpoint_path),
                            "cache": str(cache_path),
                            "relative_waveform_error": float(error.cpu()),
                            "initial_corrected_fidelity": float(
                                cached_result.components["corrected_fidelity"].cpu()
                            ),
                        }
                        print(
                            "Selected warm start: cached direct Fourier fit "
                            f"(waveform_error={float(error.cpu()):.3e}, "
                            f"F_corr={float(cached_result.components['corrected_fidelity'].cpu()):.10f})"
                        )
                        return cached_raw
                except Exception as exc:
                    print(f"[warning] Ignoring incompatible direct-fit cache: {exc}")

            initial_raw = None
            if (
                checkpoint_parameterization in {"direct_fourier", "bounded_fourier", ""}
                and checkpoint is not None
                and checkpoint.get("params") is not None
            ):
                physical = torch.as_tensor(
                    checkpoint["params"], dtype=self.real_dtype
                ).reshape(-1)
                if physical.numel() == self.pulse.bounds.n_parameters:
                    try:
                        initial_raw = self.pulse.raw_from_physical(physical).to(self.device)
                    except Exception:
                        initial_raw = None

            fit = self.pulse.fit_to_waveform(
                old_grid,
                old_mw,
                steps=self.config.warm_start.fit_steps,
                learning_rate=self.config.warm_start.learning_rate,
                gradient_clip_norm=self.config.warm_start.gradient_clip_norm,
                initial_raw=initial_raw,
                restarts=self.config.warm_start.fit_restarts,
                lbfgs_steps=self.config.warm_start.fit_lbfgs_steps,
                seed=self.config.seed,
            )
            fit_raw = fit.raw.to(self.device)
            with torch.no_grad():
                fit_result, _, _ = self.evaluate(fit_raw, old_grid)
            fit_fidelity = float(
                fit_result.components["corrected_fidelity"].detach().cpu()
            )
            acceptable_waveform = (
                fit.relative_error <= self.config.warm_start.warning_relative_error
            )
            acceptable_gate = (
                self.config.warm_start.accept_imperfect_fit
                and fit_fidelity
                >= self.config.warm_start.minimum_corrected_fidelity
                and fit.peak_fraction <= 1.0 + 1.0e-4
            )
            print(
                "Direct Fourier compression: "
                f"waveform_error={fit.relative_error:.3e}, "
                f"peak/bound={fit.peak_fraction:.4f}, "
                f"F_corr={fit_fidelity:.10f}, "
                f"phi_ABC={float(fit_result.coordinates['ABC'].detach().cpu()):+.6f}"
            )
            if not (acceptable_waveform or acceptable_gate):
                print(
                    "[warning] The direct Fourier compression failed both the waveform "
                    "and gate-fidelity acceptance criteria; using a seeded direct "
                    "Fourier initialization instead."
                )
                return random_initialization(
                    "seeded_random_after_failed_direct_compression"
                )

            self.warm_start_info = {
                "source": "direct_fourier_compression",
                "checkpoint": str(checkpoint_path),
                "relative_waveform_error": fit.relative_error,
                "initial_relative_waveform_error": fit.initial_relative_error,
                "peak_fraction_of_bound": fit.peak_fraction,
                "initial_corrected_fidelity": fit_fidelity,
                "initial_coordinates": {
                    key: float(value.detach().cpu())
                    for key, value in fit_result.coordinates.items()
                },
            }
            if self.config.warm_start.cache_fit:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "params_raw": fit_raw.detach().cpu(),
                        "waveform_fingerprint": fingerprint,
                        "basis_size": self.config.basis_size,
                        "bounds": {
                            "max_field_uT": self.pulse.bounds.max_field_uT,
                            "min_angular_frequency": self.pulse.bounds.min_angular_frequency,
                            "max_angular_frequency": self.pulse.bounds.max_angular_frequency,
                            "phase_bound": self.pulse.bounds.phase_bound,
                            "taper_fraction": self.pulse.bounds.taper_fraction,
                        },
                        "source_checkpoint": str(checkpoint_path),
                        "fit": self.warm_start_info,
                    },
                    cache_path,
                )
                self.warm_start_info["cache"] = str(cache_path)
            print("Selected warm start: direct Fourier compression of saved waveform")
            return fit_raw

        # Compatibility path for legacy bounded-Fourier runs.
        if checkpoint is not None and checkpoint.get("params") is not None:
            physical = torch.as_tensor(
                checkpoint["params"], dtype=self.real_dtype
            ).reshape(-1)
            if physical.numel() == self.pulse.bounds.n_parameters:
                try:
                    raw = self.pulse.raw_from_physical(physical).to(self.device)
                    self.warm_start_info = {
                        "source": "legacy_physical_params",
                        "checkpoint": str(checkpoint_path),
                    }
                    print("Selected warm start: legacy physical Fourier parameters")
                    return raw
                except Exception as exc:
                    print(f"[warning] Could not convert legacy physical parameters: {exc}")

        if old_grid is not None and old_mw is not None and self.config.warm_start.enabled:
            try:
                fit = self.pulse.fit_to_waveform(
                    old_grid,
                    old_mw,
                    steps=self.config.warm_start.fit_steps,
                    learning_rate=self.config.warm_start.learning_rate,
                    gradient_clip_norm=self.config.warm_start.gradient_clip_norm,
                    restarts=self.config.warm_start.fit_restarts,
                    lbfgs_steps=self.config.warm_start.fit_lbfgs_steps,
                    seed=self.config.seed,
                )
                if fit.relative_error <= self.config.warm_start.warning_relative_error:
                    self.warm_start_info = {
                        "source": "legacy_drive_fit",
                        "checkpoint": str(checkpoint_path),
                        "relative_waveform_error": fit.relative_error,
                    }
                    print(
                        "Selected warm start: fitted bounded Fourier waveform "
                        f"(waveform_error={fit.relative_error:.3e})"
                    )
                    return fit.raw.to(self.device)
            except Exception as exc:
                print(f"[warning] Could not fit saved drive: {exc}")

        print(
            f"[warning] {result_dir} has no compatible warm start for {requested}; "
            "using a seeded initialization."
        )
        return random_initialization("seeded_random_after_incompatible_checkpoint")

    def _run_adam(self, raw: torch.Tensor) -> torch.Tensor:
        cfg = self.config.adam
        if not cfg.enabled or cfg.steps <= 0:
            return raw

        time_grid = self._time_grid(cfg.steps_per_ns)
        parameter = raw.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([parameter], lr=cfg.learning_rate)
        best_loss = math.inf
        best_parameter = parameter.detach().clone()

        for step in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            result, _, _ = self.evaluate(parameter, time_grid)
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"Non-finite Adam loss at step {step}.")
            result.loss.backward()
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([parameter], cfg.gradient_clip_norm)
            self._record("adam", result, parameter)
            loss_value = float(result.loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_parameter = parameter.detach().clone()
            optimizer.step()

            if step == 0 or (step + 1) % 10 == 0 or step + 1 == cfg.steps:
                print(
                    f"Adam {step + 1:4d}/{cfg.steps}: "
                    f"loss={loss_value:.6e}, "
                    f"F_corr={float(result.components['corrected_fidelity'].detach().cpu()):.8f}, "
                    f"P100sum={float(result.components['population_100_sum'].detach().cpu()):.6f}"
                )
        return best_parameter

    def _run_lbfgs(self, raw: torch.Tensor) -> torch.Tensor:
        cfg = self.config.lbfgs
        if not cfg.enabled or cfg.max_iter <= 0:
            return raw

        time_grid = self._time_grid(cfg.steps_per_ns)
        parameter = raw.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [parameter],
            max_iter=cfg.max_iter,
            history_size=cfg.history_size,
            tolerance_grad=cfg.tolerance_grad,
            tolerance_change=cfg.tolerance_change,
            line_search_fn=cfg.line_search_fn,
        )
        closure_count = 0
        best_loss = math.inf
        best_parameter = parameter.detach().clone()

        def closure() -> torch.Tensor:
            nonlocal closure_count, best_loss, best_parameter
            optimizer.zero_grad(set_to_none=True)
            result, _, _ = self.evaluate(parameter, time_grid)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("Non-finite LBFGS loss.")
            result.loss.backward()
            self._record("lbfgs", result, parameter)
            loss_value = float(result.loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_parameter = parameter.detach().clone()
            closure_count += 1
            if closure_count == 1 or closure_count % 5 == 0:
                print(
                    f"LBFGS evaluation {closure_count:4d}: "
                    f"loss={loss_value:.6e}, "
                    f"F_corr={float(result.components['corrected_fidelity'].detach().cpu()):.8f}, "
                    f"P100sum={float(result.components['population_100_sum'].detach().cpu()):.6f}"
                )
            return result.loss

        optimizer.step(closure)
        return best_parameter

    def gradient_check(self, raw: torch.Tensor, *, epsilon: float = 1.0e-6, count: int = 3) -> dict[str, Any]:
        # Use a coarse grid because this is a diagnostic, not a production evaluation.
        time_grid = self._time_grid(min(self.config.adam.steps_per_ns, 0.25))
        parameter = raw.detach().clone().requires_grad_(True)
        result, _, _ = self.evaluate(parameter, time_grid)
        result.loss.backward()
        analytic = parameter.grad.detach().clone()

        indices = torch.linspace(0, parameter.numel() - 1, count).round().to(torch.long)
        checks = []
        for index in indices.tolist():
            plus = parameter.detach().clone()
            minus = parameter.detach().clone()
            plus[index] += epsilon
            minus[index] -= epsilon
            with torch.no_grad():
                f_plus = self.evaluate(plus, time_grid)[0].loss
                f_minus = self.evaluate(minus, time_grid)[0].loss
            finite_difference = (f_plus - f_minus) / (2.0 * epsilon)
            a = float(analytic[index].cpu())
            n = float(finite_difference.cpu())
            relative_error = abs(a - n) / max(1.0, abs(a), abs(n))
            checks.append(
                {
                    "index": index,
                    "analytic": a,
                    "finite_difference": n,
                    "relative_error": relative_error,
                }
            )
        return {"loss": float(result.loss.detach().cpu()), "checks": checks}

    def _apply_seeded_parameter_perturbation(self, raw: torch.Tensor) -> torch.Tensor:
        std = float(self.config.warm_start.parameter_noise_std)
        if std == 0.0:
            return raw
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + 104729)
        noise = torch.randn(
            raw.shape, dtype=raw.dtype, device=raw.device, generator=generator
        )
        perturbed = raw + std * noise
        self.warm_start_info = {
            **self.warm_start_info,
            "parameter_noise_std": std,
            "parameter_noise_seed": self.config.seed + 104729,
        }
        print(
            "Applied seeded warm-start perturbation: "
            f"std={std:g}, seed={self.config.seed + 104729}"
        )
        return perturbed

    def run(self) -> Path:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        if self.config.deterministic_algorithms:
            torch.use_deterministic_algorithms(True)

        raw = self._load_initial_raw()
        raw = self._apply_seeded_parameter_perturbation(raw)
        print(f"Device: {self.device}")
        print(f"Logical basis indices: {self.basis_indices}")
        print(f"Initial checkpoint: {self.config.resume_from or 'seeded random'}")

        raw = self._run_adam(raw)
        raw = self._run_lbfgs(raw)

        # Always report and save the final gate on the configured production
        # resolution, even when optimization used a coarser grid.
        final_grid = self._time_grid(self.config.final_steps_per_ns)
        final_result, controls, propagator = self.evaluate(raw, final_grid)
        self._record("final", final_result, raw)
        return self._save(raw, final_grid, controls, propagator, final_result)

    def _save(
        self,
        raw: torch.Tensor,
        time_grid: torch.Tensor,
        controls: list[torch.Tensor],
        propagator: torch.Tensor,
        result: ObjectiveResult,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
        configured = Path(self.config.output_dir)
        output_dir = configured if configured.is_absolute() else self.project_root / configured
        output_dir.mkdir(parents=True, exist_ok=True)

        physical = self.pulse.physical_vector(raw).detach().cpu()
        parameterization = self._parameterization_name()
        target_gate = self.objective.frame.conj().T @ self.objective.target_frame @ self.objective.frame
        with torch.no_grad():
            trajectory = self._trajectory(
                controls, time_grid, return_traces=True
            )
            peak_field_uT = float(torch.amax(torch.abs(controls[0])).detach().cpu())
            peak_fraction = peak_field_uT / self.pulse.bounds.max_field_uT
            reconstructed_raw = self.pulse.raw_from_physical(physical).to(self.device)
            reconstructed_drive = self.pulse.drive(time_grid, reconstructed_raw)
            reconstruction_error = float(
                (
                    torch.linalg.vector_norm(reconstructed_drive - controls[0])
                    / torch.clamp(
                        torch.linalg.vector_norm(controls[0]), min=1.0e-30
                    )
                ).detach().cpu()
            )
        trajectory_payload = {
            "time_s": trajectory.times.detach().cpu(),
            "population_100_trace": trajectory.population_100_trace.detach().cpu(),
            "logical_survival_trace": trajectory.logical_survival_trace.detach().cpu(),
            "population_100_sum": float(
                trajectory.population_100_sum.detach().cpu()
            ),
            "initial_logical_state": "000",
            "target_logical_state": "100",
        }
        checkpoint = {
            "params": physical,
            "params_raw": raw.detach().cpu(),
            "pulse_parameterization": parameterization,
            "reference_time_grid": (
                self.pulse.reference_time_grid.detach().cpu()
                if isinstance(self.pulse, ReferenceResidualPulse)
                else None
            ),
            "reference_mw_drive": (
                self.pulse.reference_mw_drive.detach().cpu()
                if isinstance(self.pulse, ReferenceResidualPulse)
                else None
            ),
            "fom": float(result.loss.detach().cpu()),
            "time_grid": time_grid.detach().cpu(),
            "pulse_settings": [
                {
                    "basis_type": {
                        "reference_residual_fourier": "ReferenceResidualFourier",
                        "direct_fourier": "DirectFourier",
                        "bounded_fourier": "BoundedFourier",
                    }[parameterization],
                    "basis_size": self.config.basis_size,
                    "maximal_pulse": self.pulse.bounds.max_field_uT,
                    "amplitude_unit": "microtesla",
                    "maximal_amplitude": self.pulse.bounds.component_amplitude_bound,
                    "maximal_frequency": self.pulse.bounds.max_angular_frequency,
                    "minimal_frequency": self.pulse.bounds.min_angular_frequency,
                    "maximal_phase": self.pulse.bounds.phase_bound,
                    "channel_type": "MW",
                    "taper_fraction": self.pulse.bounds.taper_fraction,
                }
            ],
            "Δ": self.delta_e,
            "drive": [control.detach().cpu() for control in controls],
            "basis_indices": list(self.basis_indices),
            "target_gate": target_gate.detach().cpu(),
            "timestamp": timestamp,
            "objective_type": "Support-Selective Gate Transformation",
            "gate": self.config.gate.upper(),
            "optimization_config": self.config.as_dict(),
            "warm_start": self.warm_start_info,
            "metrics": {
                key: float(value.detach().cpu())
                for key, value in result.components.items()
            },
            "coordinates": {
                key: float(value.detach().cpu())
                for key, value in result.coordinates.items()
            },
            "peak_field_uT": peak_field_uT,
            "peak_fraction_of_bound": peak_fraction,
            "parameter_reconstruction_relative_error": reconstruction_error,
            "manuscript_equation_exact": parameterization == "direct_fourier",
            "trajectory_metrics": trajectory_payload,
        }
        torch.save(checkpoint, output_dir / "pulse_solution.pt")
        torch.save(propagator.detach().cpu(), output_dir / "optimized_propagator.pt")
        projected = self.objective.logical_block(propagator).detach().cpu()
        torch.save(projected, output_dir / "propagator_projected.pt")
        torch.save(trajectory_payload, output_dir / "electron_trajectory_metrics.pt")
        np.savetxt(output_dir / "best_params.txt", physical.numpy())

        if parameterization == "direct_fourier":
            n = self.config.basis_size
            amplitudes = physical[:n].numpy()
            frequencies = physical[n : 2 * n].numpy()
            phases = physical[2 * n :].numpy()
            with (output_dir / "fourier_components.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "component",
                        "amplitude_uT",
                        "angular_frequency_rad_s",
                        "frequency_mhz",
                        "phase_rad",
                    ],
                )
                writer.writeheader()
                for index, (amplitude, frequency, phase) in enumerate(
                    zip(amplitudes, frequencies, phases), start=1
                ):
                    writer.writerow(
                        {
                            "component": index,
                            "amplitude_uT": float(amplitude),
                            "angular_frequency_rad_s": float(frequency),
                            "frequency_mhz": float(frequency / (2.0 * math.pi * 1.0e6)),
                            "phase_rad": float(phase),
                        }
                    )

        if self.history:
            fieldnames = sorted({key for row in self.history for key in row})
            with (output_dir / "optimization_history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)

        summary = {
            "timestamp": timestamp,
            "gate": self.config.gate.upper(),
            "output_dir": str(output_dir),
            "loss": float(result.loss.detach().cpu()),
            "metrics": checkpoint["metrics"],
            "coordinates": checkpoint["coordinates"],
            "config": self.config.as_dict(),
            "warm_start": self.warm_start_info,
            "pulse_parameterization": parameterization,
            "peak_field_uT": peak_field_uT,
            "peak_fraction_of_bound": peak_fraction,
            "parameter_reconstruction_relative_error": reconstruction_error,
            "manuscript_equation_exact": parameterization == "direct_fourier",
            "git": _git_metadata(self.project_root),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(self.device),
            },
        }
        with (output_dir / "optimization_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)

        print(f"Saved optimized checkpoint to {output_dir}")
        print(f"Final corrected fidelity: {summary['metrics']['corrected_fidelity']:.10f}")
        print(f"Final P100 sum: {summary['metrics']['population_100_sum']:.8f}")
        print(
            f"Final sampled peak: {peak_field_uT:.6f} uT "
            f"({peak_fraction:.6f} of bound)"
        )
        if parameterization == "direct_fourier":
            print(
                "Direct-Fourier reconstruction error: "
                f"{reconstruction_error:.3e}"
            )
        return output_dir
