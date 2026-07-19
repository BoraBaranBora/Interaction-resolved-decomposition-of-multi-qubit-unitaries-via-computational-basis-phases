param(
    [ValidateSet("all", "zzz", "xzz")]
    [string]$Gate = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -c "from control_optimization.pulse import DirectFourierPulse, ReferenceResidualPulse; from control_optimization.runner import ControlOptimizer; print('Control APIs OK')"
if ($LASTEXITCODE -ne 0) { throw "Control API import check failed" }

$gates = if ($Gate -eq "all") { @("zzz", "xzz") } else { @($Gate) }
foreach ($g in $gates) {
    $config = ".\configs\control_${g}_noise_robust_smoke.json"
    Write-Host "`n=== $($g.ToUpper()) smoke optimization ===" -ForegroundColor Cyan
    python .\optimize_control.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "$($g.ToUpper()) smoke optimization failed" }

    if ($g -eq "zzz") {
        $reference = ".\results\control_zzz_direct"
        $robust = ".\results_smoke\control_zzz_noise_robust"
        $figure = ".\Bilder\electron_population_integral_ZZZ.png"
    } else {
        $reference = ".\results_paper\pulse_nondiagonal"
        $robust = ".\results_smoke\control_xzz_noise_robust"
        $figure = ".\Bilder\electron_population_integral_XZZ.png"
    }

    Write-Host "`n=== $($g.ToUpper()) smoke population/exposure figure ===" -ForegroundColor Green
    python -m evaluation.population_exposure `
      --gate $g `
      --reference-dir $reference `
      --robust-dir $robust `
      --steps-per-ns 0.2 `
      --output $figure
    if ($LASTEXITCODE -ne 0) { throw "$($g.ToUpper()) population figure failed" }
}
