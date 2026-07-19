## Reproducible workflows

The repository contains three independent but checkpoint-compatible workflows.

### 1. Gradient-based control optimization

```bash
python optimize_control.py --config configs/control_zzz.json
python optimize_control.py --config configs/control_xzz.json
```

The optimizer warm-starts from the published saved waveform, then applies Adam
and strong-Wolfe LBFGS to a bounded Fourier pulse. New checkpoints are written
under `results/`; reference checkpoints under `results_paper/` are not changed.

### 2. Noiseless evaluation

```bash
python -m evaluation.run --gate diagonal
python -m evaluation.run --gate nondiagonal
python -m evaluation.summarize --result-dir results/control_zzz_refined
```

### 3. Colored-dephasing ensembles

```bash
python -m noise.run --gate diagonal --channel electron:ramsey:10:50 --n-realizations 128
python -m noise.sweep --config configs/ou_electron_sweep.json
python -m noise.aggregate --root results_ou/electron_ramsey_grid_N128 --figure-dir Bilder
```

See `CONTROL_OPTIMIZATION_GUIDE.md`, `OU_SWEEP_GUIDE.md`, and
`REPOSITORY_LAYOUT.md` for details.
