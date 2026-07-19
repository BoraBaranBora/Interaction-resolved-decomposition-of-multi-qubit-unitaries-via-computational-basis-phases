param(
    [ValidateSet("all", "zzz", "xzz")]
    [string]$Gate = "all",
    [double]$StepsPerNs = 1.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$gates = if ($Gate -eq "all") { @("zzz", "xzz") } else { @($Gate) }

foreach ($g in $gates) {
    if ($g -eq "zzz") {
        $reference = ".\results\control_zzz_direct"
        $robust = ".\results\control_zzz_noise_robust"
        $output = ".\Bilder\electron_population_integral_ZZZ.png"
    } else {
        $reference = ".\results_paper\pulse_nondiagonal"
        $robust = ".\results\control_xzz_noise_robust"
        $output = ".\Bilder\electron_population_integral_XZZ.png"
    }
    python -m evaluation.population_exposure `
      --gate $g `
      --reference-dir $reference `
      --robust-dir $robust `
      --steps-per-ns $StepsPerNs `
      --output $output
    if ($LASTEXITCODE -ne 0) { throw "$($g.ToUpper()) population figure failed" }
}
