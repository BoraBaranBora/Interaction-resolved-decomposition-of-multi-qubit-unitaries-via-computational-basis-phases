# Reproducible OU electron-dephasing sweep

The grid is defined in `configs/ou_electron_sweep.json`. The runner uses the
same Python interpreter as the active environment, calls
`reproduce_dephasing.py` once per grid point, records a log for each run, and
writes `sweep_manifest.json` with the Git commit, dirty-tree flag, Python and
package versions, commands, parameters, and run statuses.

## Validate the experiment without running it

```powershell
python .\scripts\run_ou_sweep.py `
  --config .\configs\ou_electron_sweep.json `
  --dry-run
```

## Run the 128-realization grid

```powershell
python .\scripts\run_ou_sweep.py `
  --config .\configs\ou_electron_sweep.json
```

The default output directory is:

```text
results_ou\electron_ramsey_grid_N128
```

Interrupted runs are resumed automatically because completed points are
identified by their `ensemble_summary.json` file.

## Aggregate and make the manuscript figures

```powershell
python .\scripts\aggregate_ou_sweep.py `
  --root .\results_ou\electron_ramsey_grid_N128 `
  --figure-dir .\Bilder
```

This creates:

```text
results_ou\electron_ramsey_grid_N128\sweep_summary.csv
results_ou\electron_ramsey_grid_N128\sweep_summary.json
Bilder\ou_fidelity_sweep_ZZZ.png
Bilder\ou_fidelity_sweep_XZZ.png
Bilder\ou_infidelity_comparison.png
```

## Production rerun at 512 realizations

The checked-in config remains unchanged. Override only the ensemble size:

```powershell
python .\scripts\run_ou_sweep.py `
  --config .\configs\ou_electron_sweep.json `
  --n-realizations 512

python .\scripts\aggregate_ou_sweep.py `
  --root .\results_ou\electron_ramsey_grid_N512 `
  --figure-dir .\Bilder
```

## Files to commit

Commit the runner, aggregator, and JSON configuration. Raw Monte Carlo outputs
under `results_ou/` should remain ignored. Commit final manuscript figures only
once the production sweep and convergence checks are complete.
