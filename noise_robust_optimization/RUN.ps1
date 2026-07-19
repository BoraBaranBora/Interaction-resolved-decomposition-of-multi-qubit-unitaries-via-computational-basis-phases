param(
    [ValidateSet("all", "zzz", "xzz")]
    [string]$Gate = "all",
    [double]$MinimumFidelity = 0.9970,
    [switch]$OverwriteSelection
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& .\scripts\RUN_ROBUST_CONTINUATION.ps1 `
  -Gate $Gate `
  -MinimumFidelity $MinimumFidelity `
  -OverwriteSelection:$OverwriteSelection
