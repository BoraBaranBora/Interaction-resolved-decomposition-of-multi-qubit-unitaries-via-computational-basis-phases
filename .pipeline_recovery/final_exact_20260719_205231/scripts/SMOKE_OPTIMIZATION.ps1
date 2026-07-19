$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python .\optimize_control.py --config .\configs\control_zzz_noise_robust_smoke.json
if ($LASTEXITCODE -ne 0) { throw "ZZZ smoke optimization failed" }

python .\optimize_control.py --config .\configs\control_xzz_noise_robust_smoke.json
if ($LASTEXITCODE -ne 0) { throw "XZZ smoke optimization failed" }

python -m evaluation.population_exposure `
  --gate zzz `
  --reference-dir .\results\control_zzz_direct `
  --robust-dir .\results_smoke\control_zzz_noise_robust `
  --steps-per-ns 0.2 `
  --output .\Bilder\electron_population_integral_ZZZ.png

python -m evaluation.population_exposure `
  --gate xzz `
  --reference-dir .\results_paper\pulse_nondiagonal `
  --robust-dir .\results_smoke\control_xzz_noise_robust `
  --steps-per-ns 0.2 `
  --output .\Bilder\electron_population_integral_XZZ.png
