$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -m evaluation.population_exposure `
  --gate zzz `
  --reference-dir .\results\control_zzz_direct `
  --robust-dir .\results\control_zzz_noise_robust `
  --steps-per-ns 1.0 `
  --output .\Bilder\electron_population_integral_ZZZ.png

python -m evaluation.population_exposure `
  --gate xzz `
  --reference-dir .\results_paper\pulse_nondiagonal `
  --robust-dir .\results\control_xzz_noise_robust `
  --steps-per-ns 1.0 `
  --output .\Bilder\electron_population_integral_XZZ.png
