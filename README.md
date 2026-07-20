# Dephasing-robust tripartite gates in an NV-center spin register

This repository contains the three-stage numerical workflow used to synthesize
and evaluate single-pulse \(ZZZ\) and \(XZZ\) tripartite entangling interactions
in an NV electron--nuclear spin register.

The workflow follows the structure of the manuscript:

1. support-selective tripartite interaction synthesis;
2. excitation-minimized pulse refinement;
3. Ornstein--Uhlenbeck electron-spin dephasing analysis.

## Repository workflow

### Stage 1: support-selective tripartite synthesis

Stage 1 optimizes the \(ZZZ\) and \(XZZ\) interactions with the population
penalty disabled. The objective fixes the tripartite phase, suppresses pairwise
phases, and leaves one-body phases available for a calibrated local correction.

The existing pre-population checkpoints are used as warm starts. A small,
deterministic perturbation is applied before reoptimization:

```text
warm_start.parameter_noise_std = 0.002
```

This is a reproducibility run inside the known high-fidelity basin, not a new
global-search claim.

Main outputs:

```text
results_smoke/stage1_local_equivalence/zzz
results_smoke/stage1_local_equivalence/xzz

results/stage1_local_equivalence/zzz
results/stage1_local_equivalence/xzz

Bilder/stage1_ZZZ_pulse.png
Bilder/stage1_XZZ_pulse.png
Bilder/stage1_ZZZ_phase_trajectories.png
Bilder/stage1_XZZ_phase_trajectories.png
```

Stage 1 plotting conventions:

- pulse amplitudes are plotted as
  \(\Omega(t)/(2\pi)\) in MHz;
- the \(ZZZ\) tripartite reference line is at \(-\pi/4\);
- the \(XZZ\) tripartite reference line is at \(+\pi/4\).

### Stage 2: excitation-minimized refinement

Each Stage 1 checkpoint is continued without another random perturbation. The
refinement minimizes the sampled logical population

```text
population_100_sum = sum_k P_100(t_k), initial state |000>
```

while retaining the Stage 1 interaction objective and corrected-fidelity
constraint.

Continuation weights:

```text
1e-4, 3e-4, 1e-3
```

The lowest-population candidate that satisfies the requested fidelity threshold
is copied to the canonical Stage 2 output directory.

Main outputs:

```text
results_smoke/stage2_population_refinement/zzz
results_smoke/stage2_population_refinement/xzz

results/stage2_population_refinement/zzz
results/stage2_population_refinement/xzz

Bilder/stage2_ZZZ_population_before.png
Bilder/stage2_ZZZ_population_after.png
Bilder/stage2_XZZ_population_before.png
Bilder/stage2_XZZ_population_after.png

generated/stage2_population_comparison.csv
generated/stage2_population_comparison.tex
```

### Stage 3: colored-dephasing evaluation

The final Stage 2 controls are propagated under stationary
Ornstein--Uhlenbeck electron-spin detuning.

Cartesian grid:

```text
T2*   = 2, 5, 10, 20, 50 us
tau_c = 0.1, 0.3, 1, 3, 15, 30 us
```

The evaluation also includes the three material scenarios used in the
manuscript.

Main outputs:

```text
results_ou/

Bilder/ou_fidelity_sweep_ZZZ.png
Bilder/ou_fidelity_sweep_XZZ.png

generated/experimental_scenario_results.csv
generated/experimental_scenario_results.tex
```

Stage 3 writes a manifest into each sweep output directory. Completed points
are detected and skipped when a run is resumed.

Do not run two sweeps against the same output directory at the same time. On
Windows, also avoid opening `sweep_manifest.json` in an editor or preview pane
while the sweep is active, because another process can temporarily lock the
file.

## Environment setup

Run all commands from the repository root.

Create a virtual environment:

```powershell
py -3.10 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the repository requirements using the dependency file supplied by the
project:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repository uses another environment file, such as `pyproject.toml`,
install from that file instead.

## Smoke run

Run the complete low-cost smoke workflow before launching production:

```powershell
python .\scripts\run_three_stage_pipeline.py --mode smoke --overwrite
```

The smoke run is intended to verify:

- checkpoint loading;
- optimization entry points;
- Stage 1 pulse and phase figures;
- Stage 2 population refinement and generated tables;
- OU propagation, manifest writing, and plotting.

Smoke values are not publication results.

The stages can also be run separately:

```powershell
python .\scripts\stage1_local_equivalence.py --mode smoke --gate all --overwrite
python .\scripts\stage2_population_refinement.py --mode smoke --gate all --minimum-fidelity 0.997 --overwrite
python .\scripts\stage3_noise_evaluation.py --mode smoke
```

## Publication run

After the smoke outputs have been inspected, launch the complete production
workflow:

```powershell
python .\scripts\run_three_stage_pipeline.py --mode publication --overwrite
```

Use `--overwrite` only when Stage 1 and Stage 2 should be regenerated. Stage 3
is resumable because completed sweep points are skipped.

To continue only Stage 3 after an interruption:

```powershell
python .\scripts\stage3_noise_evaluation.py --mode publication
```

Do not start a second publication process while the first one is active.

## Publication-output checks

Before using the results in the manuscript, verify the following.

### Stage 1

- both gate optimizations completed without a traceback;
- the pulse axes show \(\Omega(t)\,[2\pi\times\mathrm{MHz}]\);
- the \(ZZZ\) phase reference is \(-\pi/4\);
- the \(XZZ\) phase reference is \(+\pi/4\);
- terminal tripartite and pairwise coordinates are consistent with the target;
- corrected fidelities and logical survival are recorded.

### Stage 2

- the selected checkpoint satisfies the corrected-fidelity threshold;
- \(S_{100}\) and \(I_{100}\) are reduced relative to Stage 1;
- before/after population plots use the same normalization;
- the generated CSV and LaTeX table agree with the selected checkpoints.

### Stage 3

- every Cartesian and material point is marked complete;
- both OU sweep figures were regenerated;
- standard errors and realization statistics are present;
- the noiseless or weak-noise limit approaches the independently evaluated
  control floor;
- no output directory contains an abandoned `*.tmp` manifest.

## Git and generated data

Source code, configurations, compact generated tables, and final manuscript
figures may be committed when they are part of the reproducibility record.

Large raw trajectory ensembles and temporary sweep products should normally
remain outside Git and be archived separately, for example on Zenodo.

Recommended ignore rules include:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

results_ou/
generated/stage3_configs/

*.tmp
```

Do not merge the feature branch into `main` until the publication run has
completed and the output checklist above has been reviewed.

## Branch workflow

Push the working branch:

```powershell
git push
```

After the publication run, inspect the final changes:

```powershell
git status --short
git diff
```

Commit only the intended source, configuration, documentation, compact tables,
and final figures:

```powershell
git add README.md .gitignore evaluation scripts noise generated Bilder
git status --short
git diff --cached --stat
git commit -m "Finalize tripartite-gate production workflow and results"
git push
```

Review the feature branch through a pull request before merging it into `main`.

## Data and manuscript integration

The generated LaTeX tables are designed to be included by the manuscript:

```text
generated/stage2_population_comparison.tex
generated/experimental_scenario_results.tex
```

The final raw numerical archive should contain the production configurations,
selected checkpoints, manifests, compact summaries, and enough metadata to
reproduce every manuscript figure and table.
