param(
    [ValidateSet("all", "zzz", "xzz")]
    [string]$Gate = "all"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& .\scripts\SMOKE_OPTIMIZATION.ps1 -Gate $Gate
