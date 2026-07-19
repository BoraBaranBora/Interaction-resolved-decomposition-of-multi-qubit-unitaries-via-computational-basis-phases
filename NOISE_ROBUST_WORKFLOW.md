# Noise-robust tripartite-gate workflow

This patch implements the staged analysis used by the manuscript
`nv_tripartite_noise_robust_preparation.tex`.

## Scientific logic

1. **Reference controls.** Keep the validated direct 11-component Fourier
   pulses as immutable high-fidelity reference checkpoints.
2. **Use the open local sector.** The support-selective target fixes the
   tripartite coefficient and suppresses pairwise terms, but it does not fix the
   one-body phase coordinates.
3. **Penalize electronic dephasing exposure.** Warm-start from the reference
   controls and add

   \[
   J_e = \frac{1}{8T}\sum_x\int_0^T
   \left[1-\langle Z_A\rangle_x(t)^2\right]dt.
   \]

   This is the logical-basis average of `Var(Z_A)` along the noiseless
   trajectories. It is differentiable and is not fixed by unitarity.
4. **Diagnose trajectories.** Plot both `Var(Z_A)` and the population in the
   electron manifold opposite to each input basis state. The population plot is
   intuitive; the OU gate calculation is the actual robustness validation.
5. **Paired noise comparison.** Evaluate reference and robust pulses with common
   random numbers on the exact Cartesian product

   - `T2* = [2, 5, 10, 20, 50] us`
   - `tau_c = [0.1, 0.3, 1, 3, 15, 30] us`.

6. **Experimental overlays.** Run Bar-Gill/Walsworth, Hayashi, and
   Bauch/Walsworth source-consistent scenarios separately and overlay their
   actual simulated fidelities on each gate's sweep figure.
7. **Interpretation.** Curves flattening toward the noiseless baseline at large
   `T2*` is the generic weak-noise limit. Robustness created by the optimization
   is established only by a lower stochastic loss for the robust pulse than for
   the reference pulse at the same noise point.

## Why raw excited-state population is not the cost

The electron is itself a logical qubit. Averaging the population of one electron
manifold over all eight logical basis states is fixed at one half under unitary
evolution, so that average cannot be minimized. The implemented cost instead
penalizes electron-Z variance and reports *opposite-manifold excursion relative
to each input state*.

## Step 1: install and test

```powershell
Expand-Archive `
  "$HOME\Downloads\nv_noise_robust_tripartite_pipeline.zip" `
  "$HOME\Downloads\nv_noise_robust_patch" `
  -Force

$Patch = "$HOME\Downloads\nv_noise_robust_patch\nv_noise_robust_tripartite_pipeline"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "$Patch\INSTALL_NOISE_ROBUST_PIPELINE.ps1"

python -m pytest -q
```

The unit suite tests the support-selective objective, bounded Fourier pulse,
OU calibration, Cartesian-grid configuration, and trajectory exposure
functional. The first repository optimization evaluation additionally checks
that the new trajectory propagator reproduces the repository's existing
`evolution.get_propagator` to relative Frobenius error below `1e-9`.

## Step 2: smoke optimization and preliminary PNGs

```powershell
& .\scripts\SMOKE_PIPELINE.ps1
```

This deliberately uses very few optimizer iterations, four OU realizations per
Cartesian point, and eight per experimental scenario. It produces manuscript-
compatible preliminary files:

```text
Bilder/electron_population_integral_ZZZ.png
Bilder/electron_population_integral_XZZ.png
Bilder/ou_fidelity_sweep_ZZZ_reference_vs_robust.png
Bilder/ou_fidelity_sweep_XZZ_reference_vs_robust.png
generated/experimental_scenario_results.tex
```

Smoke values are not publication results.

## Step 3: full robust optimization

```powershell
& .\scripts\RUN_ROBUST_CONTINUATION.ps1 `
  -Gate all `
  -MinimumFidelity 0.9970 `
  -OverwriteSelection
```

For each gate the script continues through exposure weights `0.01`, `0.03`, and
`0.10`, always warm-starting from the preceding solution. It then selects the
lowest-exposure candidate above the fidelity threshold and copies it to:

```text
results/control_zzz_noise_robust
results/control_xzz_noise_robust
```

Inspect `robust_selection.json` before treating the selected checkpoint as
final. A publication pulse must also pass a finer propagation-grid check and a
peak-amplitude audit.

Generate the final trajectory figures with:

```powershell
& .\scripts\MAKE_POPULATION_FIGURES.ps1
```

## Step 4: overnight noise analysis

```powershell
& .\scripts\RUN_OVERNIGHT_NOISE.ps1
```

The overnight configuration uses 64 trajectories per Cartesian point and 256
per experimental scenario. It evaluates both reference and robust pulses and
uses identical seeds for paired comparisons.

## Step 5: publication noise analysis

```powershell
& .\scripts\RUN_PUBLICATION_NOISE.ps1
```

The publication configuration uses 256 trajectories per Cartesian point and
512 per experimental scenario at one propagation sample per nanosecond. It
writes the same figure names as the smoke run, replacing preliminary outputs.

## Acceptance criteria

Do not claim control-induced noise robustness unless all hold:

1. corrected noiseless fidelity exceeds the chosen threshold;
2. the support coordinates and diagonality remain within the reference
   tolerances;
3. `J_e` decreases and the population-excursion plots change consistently;
4. the robust pulse has lower dephasing-induced fidelity loss than the reference
   pulse over a meaningful portion of the paired OU grid;
5. the result is not based only on flattening at large `T2*`;
6. at least the experimental-scenario table shows minimal or moderate loss for
   the selected robust pulse;
7. propagation-grid refinement and peak-amplitude checks pass.
