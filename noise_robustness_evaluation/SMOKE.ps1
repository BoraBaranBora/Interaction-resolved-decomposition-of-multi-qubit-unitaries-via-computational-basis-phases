$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& .\scripts\SMOKE_NOISE_AND_FIGURES.ps1
