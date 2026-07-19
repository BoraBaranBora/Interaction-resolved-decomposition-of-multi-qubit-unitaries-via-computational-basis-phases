$ErrorActionPreference = "Stop"

$Required = @(
    ".\src\quantum_model_NV.py",
    ".\src\evolution.py",
    ".\reproduce_numerics.py"
)
foreach ($Path in $Required) {
    if (-not (Test-Path $Path)) {
        throw "Run this script from the repository root; missing $Path"
    }
}

$Rules = @'

# Gradient-based control optimization
!optimize_control.py
!control_optimization/
!control_optimization/*.py
!evaluation/
!evaluation/*.py
!noise/
!noise/*.py
!legacy/
!legacy/*.md
!configs/control_*.json
!tests/test_control_*.py
!tests/test_fourier_pulse.py
!tests/test_config.py

# Generated optimization outputs
results/control_*_refined/
'@

$GitIgnore = Get-Content .\.gitignore -Raw
if ($GitIgnore -notmatch "Gradient-based control optimization") {
    Add-Content .\.gitignore $Rules
    Write-Host "Updated .gitignore"
} else {
    Write-Host ".gitignore already contains control rules"
}

Write-Host "Running control tests..."
python -m pytest -q `
    .\tests\test_control_objective.py `
    .\tests\test_fourier_pulse.py `
    .\tests\test_config.py
if ($LASTEXITCODE -ne 0) {
    throw "Control tests failed"
}

Write-Host "Installation checks passed."
Write-Host "Next: python .\optimize_control.py --config .\configs\control_zzz.json --gradient-check"
