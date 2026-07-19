# Part I — Noise-robust optimization

This part refines the existing direct-Fourier ZZZ and XZZ checkpoints. It does
not refit or replace the reference pulse before optimization: when the saved
checkpoint uses `direct_fourier`, `params_raw` is loaded exactly.

The terminal gate objective is unchanged. The added trajectory term is the
normalized time integral of the logical-basis-averaged electron-Z variance.
The opposite-electron-manifold population is reported as a separate diagnostic.

Reference checkpoints expected by the scripts:

- ZZZ: `results/control_zzz_direct/pulse_solution.pt`
- XZZ: `results_paper/pulse_nondiagonal/pulse_solution.pt`

Run the smoke test first:

```powershell
& .\noise_robust_optimization\SMOKE.ps1 -Gate all
```

Then run the continuation:

```powershell
& .\noise_robust_optimization\RUN.ps1 -Gate all -MinimumFidelity 0.9970
```

The continuation uses exposure weights `0.01`, `0.03`, and `0.10`, with each
stage warm-started from the previous stage. The canonical selected checkpoints
are written to `results/control_zzz_noise_robust` and
`results/control_xzz_noise_robust`.
