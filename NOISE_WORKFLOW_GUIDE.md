# Three realistic NV-material OU scenarios

## Scope

The production calculation is deliberately limited to electronic longitudinal
detuning noise,

```text
H_noise(t) = beta(t) * (Z_A / 2 tensor I_nuclear).
```

No nuclear-noise channel, refocusing pulse, Hahn echo, or dynamical-decoupling
sequence is inserted into a gate. Noise-spectroscopy and echo measurements are
used only to infer environmental parameters before propagation.

There is no Cartesian parameter grid. The code runs exactly three material
settings for each gate:

1. `bargill_12c_cvd`: directly fitted Lorentzian bath parameters for a purified
   12C CVD sample (`Delta=30(10) kHz`, `tau_c=10(5) us`).
2. `hayashi_hpht_no8`: rounded same-sample fit for HPHT sample No. 8
   (`lambda≈0.025 MHz`, `tau_c≈14 us`). The source Hamiltonian is
   `H_I=lambda f(t) sigma_z`; under the propagation convention
   `H_noise=beta(t) Z/2`, this gives `sigma_beta/(2 pi)=50 kHz`.
3. `bauch_12c_2ppm`: an engineered 99.95%+ 12C benchmark at 2 ppm nitrogen,
   derived from the measured concentration laws for Ramsey `T2*` and echo `T2`.

The third scenario is explicitly labeled as a concentration-scaled benchmark,
not as a separately tabulated sample. The resolved central values used by the
code are recorded in `resolved_material_scenarios.json`.

## Pulse checkpoints

```text
ZZZ: results/control_zzz_direct/pulse_solution.pt
XZZ: results_paper/pulse_nondiagonal/pulse_solution.pt
```

Both controls use the direct 11-component Fourier representation.

## Install and validate

```powershell
python -m pytest `
  .\tests\test_ou_ramsey.py `
  .\tests\test_noise_model.py `
  .\tests\test_noise_sweep_config.py `
  -q

python -m noise.validate `
  --config .\configs\ou_three_materials.json
```

## Smoke study

```powershell
python -m noise.sweep `
  --config .\configs\ou_three_materials_smoke.json

python -m noise.aggregate `
  --root .\results_ou\electron_ou_three_realistic_materials_smoke_N8 `
  --figure-dir .\Bilder
```

## Production study

The production configuration contains six conditions: two gates times three
material settings. It is resumable and defaults to 256 trajectories per
condition.

```powershell
python -m noise.sweep `
  --config .\configs\ou_three_materials.json

python -m noise.aggregate `
  --root .\results_ou\electron_ou_three_realistic_materials_N256 `
  --figure-dir .\Bilder
```

For 512 trajectories:

```powershell
python -m noise.sweep `
  --config .\configs\ou_three_materials.json `
  --n-realizations 512
```

## Generated manuscript outputs

```text
Bilder/ou_fidelity_materials_ZZZ.png
Bilder/ou_fidelity_materials_XZZ.png
Bilder/ou_dephasing_loss_three_materials.png
results_ou/.../three_material_results.csv
```

## Convergence check

Repeat all six conditions at two steps/ns by copying the production JSON and
changing `propagation_steps_per_ns` from `1.0` to `2.0`. Keep the same seed and
trajectory count so changes predominantly reflect the propagation grid.
