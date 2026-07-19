$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -m noise.sweep --config .\configs\ou_cartesian_reference_overnight.json
python -m noise.sweep --config .\configs\ou_cartesian_robust_overnight.json
python -m noise.sweep_scenarios --config .\configs\ou_experimental_reference_publication.json --n-realizations 256
python -m noise.sweep_scenarios --config .\configs\ou_experimental_robust_publication.json --n-realizations 256

python -m noise.aggregate_comparison `
  --reference-grid-root .\results_ou\electron_ou_cartesian_reference_overnight_N64 `
  --robust-grid-root .\results_ou\electron_ou_cartesian_robust_overnight_N64 `
  --reference-scenario-root .\results_ou\electron_ou_experimental_reference_publication_N256 `
  --robust-scenario-root .\results_ou\electron_ou_experimental_robust_publication_N256 `
  --figure-dir .\Bilder `
  --table-dir .\generated
