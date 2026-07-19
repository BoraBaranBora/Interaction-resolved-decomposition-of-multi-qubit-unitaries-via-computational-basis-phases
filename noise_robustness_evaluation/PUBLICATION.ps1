$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& .\scripts\RUN_PUBLICATION_NOISE.ps1
