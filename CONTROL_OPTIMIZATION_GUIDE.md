# Gradient-based control optimization

## Production workflow

The optimizer refines the published ZZZ and XZZ pulses with a differentiable,
peak-bounded Fourier ansatz. It uses:

1. verified import of the published physical parameter vector, with a saved-
   waveform fit as a fallback;
2. coarse-grid Adam exploration;
3. strong-Wolfe LBFGS refinement;
4. a final production-grid evaluation and compatible checkpoint save.

The saved control arrays are in **microtesla**, matching the numerical convention
used by `src/quantum_model_NV.py`. Frequencies are angular frequencies in rad/s
and time grids are in seconds.

## Install

Extract the bundle into the repository root. Append `.gitignore.additions` to the
repository `.gitignore` if the broad historical `*.py` rule is still present.

The historical uploaded optimization scripts should be moved to `legacy/` and
kept only for provenance. They are not imported by the new workflow.

## Test the standalone mathematics

```powershell
python -m pytest -q `
  .\tests\test_control_objective.py `
  .\tests\test_fourier_pulse.py `
  .\tests\test_config.py
```

## Check autograd through the actual NV model

```powershell
python .\optimize_control.py `
  --config .\configs\control_zzz.json `
  --gradient-check

python .\optimize_control.py `
  --config .\configs\control_xzz.json `
  --gradient-check
```

This compares selected autograd entries with central finite differences on a
coarse time grid. Large disagreement at one phase-branch crossing can be local;
consistent disagreement across parameters indicates a real integration problem.

## Refine the published pulses

```powershell
python .\optimize_control.py --config .\configs\control_zzz.json
python .\optimize_control.py --config .\configs\control_xzz.json
```

The default configs do not overwrite the reference data:

```text
results/control_zzz_refined/
results/control_xzz_refined/
```

At startup, the optimizer reports two diagnostics:

- the exact published waveform evaluated under the new objective;
- the bounded-Fourier fit used as the gradient warm start.

The exact published fidelity should agree with the existing evaluation. If its
tripartite coordinate has the opposite sign, change `target_angle_rad` from
`+pi/4` to `-pi/4` before optimizing. A poor waveform-fit error means the new
basis/bounds need adjustment; do not mistake that for a failure of the saved
pulse.

## Evaluate a refined checkpoint

```powershell
python -m evaluation.summarize `
  --result-dir .\results\control_zzz_refined

python -m evaluation.summarize `
  --result-dir .\results\control_xzz_refined
```

The optimizer saves the fields used by the existing repository:

- `drive`
- `time_grid`
- `Δ`
- `basis_indices`
- `params`
- `propagator_projected.pt`

It additionally saves `params_raw`, the full propagator, optimization history,
configuration, Git metadata, environment metadata, and warm-start diagnostics.

## Terminal objective

In the identity frame for ZZZ or a Hadamard frame on the electronic qubit for
XZZ, the loss combines:

- local-corrected process infidelity;
- the pi-periodic selected-coordinate loss
  `1 - cos(2*(phi - phi_target))`;
- pairwise-coordinate suppression;
- off-diagonal population-transfer penalty;
- projected unitarity and logical-survival penalties;
- weak fluence and smoothness regularization.

Single-qubit phase coordinates remain free in the target definition and are
removed analytically when computing the corrected process fidelity.

## Acceptance criteria before replacing a reference pulse

1. The exact saved waveform is reproduced by the old evaluation.
2. The final 1 sample/ns evaluation improves the corrected process fidelity.
3. A second, finer propagation grid agrees within the stated numerical error.
4. Pairwise coordinates and off-diagonal transfer remain suppressed.
5. The new checkpoint passes the full OU sweep using the same noise seeds.
6. Reference checkpoints remain unchanged until the replacement is documented.
