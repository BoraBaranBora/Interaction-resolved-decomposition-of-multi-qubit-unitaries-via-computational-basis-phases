$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -m noise.sweep --config .\configs\ou_cartesian_reference_smoke.json
python -m noise.sweep --config .\configs\ou_cartesian_robust_smoke.json
python -m noise.sweep_scenarios --config .\configs\ou_experimental_reference_smoke.json
python -m noise.sweep_scenarios --config .\configs\ou_experimental_robust_smoke.json

python -m noise.aggregate_comparison `
  --reference-grid-root .\results_ou\electron_ou_cartesian_reference_smoke_N4 `
  --robust-grid-root .\results_ou\electron_ou_cartesian_robust_smoke_N4 `
  --reference-scenario-root .\results_ou\electron_ou_experimental_reference_smoke_N8 `
  --robust-scenario-root .\results_ou\electron_ou_experimental_robust_smoke_N8 `
  --figure-dir .\Bilder `
  --table-dir .\generated
