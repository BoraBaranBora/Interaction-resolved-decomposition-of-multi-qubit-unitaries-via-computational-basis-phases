$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& .\scripts\SMOKE_OPTIMIZATION.ps1
& .\scripts\SMOKE_NOISE_AND_FIGURES.ps1

Write-Host "`nSmoke pipeline complete." -ForegroundColor Green
Write-Host "Preliminary figures:" 
Write-Host "  Bilder/electron_population_integral_ZZZ.png"
Write-Host "  Bilder/electron_population_integral_XZZ.png"
Write-Host "  Bilder/ou_fidelity_sweep_ZZZ_reference_vs_robust.png"
Write-Host "  Bilder/ou_fidelity_sweep_XZZ_reference_vs_robust.png"
