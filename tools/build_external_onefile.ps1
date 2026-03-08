$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path "build/lrc-processor-onefile") {
    Remove-Item "build/lrc-processor-onefile" -Recurse -Force
}
if (Test-Path "dist/lrc-processor-onefile.exe") {
    Remove-Item "dist/lrc-processor-onefile.exe" -Force
}

pyinstaller --clean --noconfirm lrc-processor-onefile.spec

Write-Host "Built: $Root/dist/lrc-processor-onefile.exe"
