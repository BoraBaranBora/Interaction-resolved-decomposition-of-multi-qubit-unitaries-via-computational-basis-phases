# Repository layout

The repository is organized around three scientific workflows:

```text
control_optimization/   bounded pulse, support-selective objective, Adam/LBFGS
                         runner and checkpoint creation

evaluation/             noiseless reproduction, checkpoint summaries and gate
                         diagnostics

noise/                  OU ensemble simulation, parameter sweeps and aggregation

configs/                version-controlled control and noise experiment settings
results_paper/           immutable published/reference checkpoints
results/                 newly optimized checkpoints
results_ou/              generated stochastic ensembles (not committed)
legacy/                  historical experimental optimization scripts
```

Primary commands:

```powershell
python .\optimize_control.py --config .\configs\control_zzz.json
python -m evaluation.summarize --result-dir .\results\control_zzz_refined
python -m evaluation.run --gate diagonal
python -m noise.run --gate diagonal --channel electron:ramsey:10:50 --n-realizations 128
python -m noise.sweep --config .\configs\ou_electron_sweep.json
```

The existing root reproduction and noise scripts remain available as
compatibility entry points. The package entry points delegate to them where
appropriate, so the current public results are not broken during migration.
