param([string]$Destination = (Join-Path $PSScriptRoot "../.tools/vcpkg"))
$ErrorActionPreference = "Stop"
$Baseline = "58845ed63eb19aff55e896ea1f5d51f2a0df5b66"
if (Test-Path $Destination) {
    if (-not (Test-Path (Join-Path $Destination ".git"))) {
        throw "Refusing to modify existing directory: $Destination"
    }
    $Current = & git -C $Destination rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $Current -ne $Baseline) {
        throw "Existing checkout does not match pinned baseline. Use a new directory."
    }
} else {
    & git clone https://github.com/microsoft/vcpkg.git $Destination
    if ($LASTEXITCODE -ne 0) { throw "vcpkg clone failed" }
    & git -C $Destination checkout --detach $Baseline
    if ($LASTEXITCODE -ne 0) { throw "vcpkg checkout failed" }
}
& (Join-Path $Destination "bootstrap-vcpkg.bat") -disableMetrics
if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed" }
$Resolved = (Resolve-Path $Destination).Path
Write-Host "Set the root in this PowerShell session:"
Write-Host "`$env:VCPKG_ROOT = '$Resolved'"
