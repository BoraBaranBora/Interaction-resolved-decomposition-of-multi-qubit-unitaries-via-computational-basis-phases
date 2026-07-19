# Direct-Fourier finalization stage

This patch converts a high-quality sampled teacher waveform into the literal pulse ansatz used in the manuscript,

\[
\Omega(t;\mathbf p)=\chi(t)\sum_{k=1}^{N_c}a_k\cos(\omega_k t+\phi_k).
\]

The final `direct_fourier` checkpoint contains no reference waveform, residual layer, clipping, or output rescaling. The stored amplitudes, frequencies, and phases reconstruct the stored microwave drive directly.

## ZZZ workflow

The residual-polished checkpoint remains an initialization teacher only:

```text
results/control_zzz_polish/pulse_solution.pt
```

The direct final configuration uses `N_c = 11` and writes to:

```text
results/control_zzz_direct/
```

Run the gradient check first:

```powershell
python .\optimize_control.py `
  --config .\configs\control_zzz_direct.json `
  --gradient-check
```

The first invocation performs a deterministic multi-start waveform compression and caches it at:

```text
results/control_zzz_direct/direct_fourier_warm_start.pt
```

A later invocation with the same teacher waveform and bounds reuses that cache. The diagnostic must print `Selected warm start: direct Fourier compression of saved waveform` or `cached direct Fourier fit`; it must not silently interpret residual parameters as direct coefficients.

Run the gate optimization:

```powershell
python .\optimize_control.py `
  --config .\configs\control_zzz_direct.json
```

Then verify literal agreement with the manuscript equation:

```powershell
python -m evaluation.verify_fourier `
  --result-dir .\results\control_zzz_direct
```

A valid final pulse reports:

```text
Manuscript equation exact: True
Peak constraint satisfied: True
```

Evaluate the gate metrics:

```powershell
python -m evaluation.summarize `
  --result-dir .\results\control_zzz_direct
```

## Final files

`results/control_zzz_direct/` contains:

- `pulse_solution.pt`: complete direct-Fourier checkpoint;
- `fourier_components.csv`: one row per manuscript Fourier component;
- `best_params.txt`: amplitudes, angular frequencies, and phases;
- `optimization_history.csv`: Adam and LBFGS history;
- `optimization_summary.json`: fidelity, coordinates, peak and reconstruction diagnostics;
- `optimized_propagator.pt` and `propagator_projected.pt`.

## Acceptance criteria

Use the direct checkpoint in the manuscript and OU calculations only when all of the following hold:

1. fine-grid corrected fidelity reaches the chosen threshold;
2. the tripartite coordinate remains equivalent to `pi/4` modulo `pi`;
3. pairwise coordinates remain suppressed modulo `pi`;
4. `evaluation.verify_fourier` passes;
5. the sampled peak remains below the configured six-megahertz ZZZ bound;
6. fidelity is stable when the final time-grid resolution is increased.

The residual checkpoint should remain in the repository as optimization provenance, not as the final manuscript pulse.
