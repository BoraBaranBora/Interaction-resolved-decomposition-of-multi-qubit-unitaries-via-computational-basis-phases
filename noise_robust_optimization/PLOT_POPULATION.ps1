param(
    [ValidateSet("all", "zzz", "xzz")]
    [string]$Gate = "all"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& .\scripts\MAKE_POPULATION_FIGURES.ps1 -Gate $Gate
