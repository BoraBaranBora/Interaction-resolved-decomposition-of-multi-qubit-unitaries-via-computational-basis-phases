$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $($Arguments -join ' ')"
    }
}

$required = @(
    ".\results\control_zzz_direct\pulse_solution.pt",
    ".\results_paper\pulse_nondiagonal\pulse_solution.pt",
    ".\results\control_zzz_noise_robust\pulse_solution.pt",
    ".\results\control_xzz_noise_robust\pulse_solution.pt"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) { throw "Missing checkpoint: $path" }
}

Invoke-Python -Arguments @("-m", "noise.sweep", "--config", ".\configs\ou_cartesian_reference_publication.json")
Invoke-Python -Arguments @("-m", "noise.sweep", "--config", ".\configs\ou_cartesian_robust_publication.json")
Invoke-Python -Arguments @("-m", "noise.sweep_scenarios", "--config", ".\configs\ou_experimental_reference_publication.json")
Invoke-Python -Arguments @("-m", "noise.sweep_scenarios", "--config", ".\configs\ou_experimental_robust_publication.json")
Invoke-Python -Arguments @(
    "-m", "noise.aggregate_comparison",
    "--reference-grid-root", ".\results_ou\electron_ou_cartesian_reference_publication_N256",
    "--robust-grid-root", ".\results_ou\electron_ou_cartesian_robust_publication_N256",
    "--reference-scenario-root", ".\results_ou\electron_ou_experimental_reference_publication_N512",
    "--robust-scenario-root", ".\results_ou\electron_ou_experimental_robust_publication_N512",
    "--figure-dir", ".\Bilder",
    "--table-dir", ".\generated"
)
