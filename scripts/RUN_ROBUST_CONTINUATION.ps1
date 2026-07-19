param(
    [ValidateSet("all", "zzz", "xzz")]
    [string]$Gate = "all",
    [double]$MinimumFidelity = 0.9970,
    [switch]$OverwriteSelection
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -c "from control_optimization.pulse import DirectFourierPulse, ReferenceResidualPulse; from control_optimization.runner import ControlOptimizer; print('Control APIs OK')"
if ($LASTEXITCODE -ne 0) { throw "Control API import check failed" }

$gateList = if ($Gate -eq "all") { @("zzz", "xzz") } else { @($Gate) }
$weights = @(0.01, 0.03, 0.10)

foreach ($g in $gateList) {
    $basePath = Join-Path $ProjectRoot "configs/control_${g}_noise_robust.json"
    if (-not (Test-Path $basePath)) { throw "Missing config: $basePath" }
    $base = Get-Content $basePath -Raw | ConvertFrom-Json
    $resume = if ($g -eq "zzz") { "results/control_zzz_direct" } else { "results_paper/pulse_nondiagonal" }
    if (-not (Test-Path (Join-Path $ProjectRoot "$resume/pulse_solution.pt"))) {
        throw "Missing warm-start checkpoint: $resume/pulse_solution.pt"
    }
    $candidates = @()

    foreach ($weight in $weights) {
        $slug = ("{0:0.00}" -f $weight).Replace(".", "p")
        $outputDir = "results/control_${g}_noise_robust_w${slug}"
        $candidateConfig = ($base | ConvertTo-Json -Depth 20 | ConvertFrom-Json)
        $candidateConfig.resume_from = $resume
        $candidateConfig.output_dir = $outputDir
        $candidateConfig.pulse_parameterization = "direct_fourier"
        $candidateConfig.objective_weights.electron_dephasing_exposure = $weight

        $configDir = Join-Path $ProjectRoot "results/_robust_configs"
        New-Item -ItemType Directory -Force -Path $configDir | Out-Null
        $configPath = Join-Path $configDir "control_${g}_w${slug}.json"
        $candidateConfig | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $configPath

        Write-Host "`n=== $($g.ToUpper()) exposure weight $weight ===" -ForegroundColor Cyan
        python .\optimize_control.py --config $configPath
        if ($LASTEXITCODE -ne 0) { throw "Optimization failed for $g at weight $weight" }

        $candidates += $outputDir
        $resume = $outputDir
    }

    $selectArgs = @(
        "-m", "evaluation.select_robust",
        "--minimum-fidelity", "$MinimumFidelity",
        "--output-dir", "results/control_${g}_noise_robust"
    )
    foreach ($candidate in $candidates) {
        $selectArgs += @("--candidate", $candidate)
    }
    if ($OverwriteSelection) { $selectArgs += "--overwrite" }

    Write-Host "`n=== Selecting canonical $($g.ToUpper()) robust pulse ===" -ForegroundColor Green
    python @selectArgs
    if ($LASTEXITCODE -ne 0) { throw "Selection failed for $g" }
}
