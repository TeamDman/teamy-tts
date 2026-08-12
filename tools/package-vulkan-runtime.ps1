[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ExecutablePath,

    [Parameter(Mandatory)]
    [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$executable = (Resolve-Path -LiteralPath $ExecutablePath).Path
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Executable is not a file: $executable"
}

$output = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $output | Out-Null

$executableTarget = Join-Path $output ([IO.Path]::GetFileName($executable))
Copy-Item -LiteralPath $executable -Destination $executableTarget -Force

$file = Get-Item -LiteralPath $executableTarget
$hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
$manifest = [ordered]@{
    schema_version = 1
    runtime = 'vulkan'
    feature = 'vulkan'
    python_required = $false
    native_runtime_files = @()
    external_runtime = 'Vulkan 1.3-capable graphics driver'
    validated_gpu = 'NVIDIA GeForce RTX 4090'
    model_bundle_is_separate = $true
    executable = $file.Name
    native_files = @(
        [ordered]@{
            name = $file.Name
            bytes = $file.Length
            sha256 = $hash.Hash.ToLowerInvariant()
        }
    )
}

$manifestPath = Join-Path $output 'vulkan-runtime.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Output ("Vulkan runtime package: {0}" -f $output)
Write-Output ("Executable: {0}" -f $executableTarget)
Write-Output ("Manifest: {0}" -f $manifestPath)
