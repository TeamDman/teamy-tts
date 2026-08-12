[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ExecutablePath,

    [Parameter(Mandatory)]
    [string]$LibTorchRoot,

    [Parameter(Mandatory)]
    [string]$OutputDir,

    [switch]$IncludeTorchPython
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$executable = (Resolve-Path -LiteralPath $ExecutablePath).Path
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Executable is not a file: $executable"
}

$libtorch = (Resolve-Path -LiteralPath $LibTorchRoot).Path
$libDir = Join-Path $libtorch 'lib'
if (-not (Test-Path -LiteralPath $libDir -PathType Container)) {
    throw "LibTorch lib directory is missing: $libDir"
}

$output = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $output | Out-Null

$executableTarget = Join-Path $output ([IO.Path]::GetFileName($executable))
Copy-Item -LiteralPath $executable -Destination $executableTarget -Force

$runtimeDlls = @(Get-ChildItem -LiteralPath $libDir -Filter '*.dll' -File |
    Where-Object { $IncludeTorchPython -or $_.Name -ne 'torch_python.dll' } |
    Sort-Object Name)
if ($runtimeDlls.Count -eq 0) {
    throw "No LibTorch DLLs were found in $libDir"
}

foreach ($dll in $runtimeDlls) {
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $output $dll.Name) -Force
}

$manifest = [ordered]@{
    schema_version = 1
    runtime = 'libtorch'
    python_required = $false
    torch_python_included = [bool]$IncludeTorchPython
    executable = [IO.Path]::GetFileName($executableTarget)
    native_files = @(
        foreach ($file in @(
            Get-Item -LiteralPath $executableTarget
            Get-ChildItem -LiteralPath $output -Filter '*.dll' -File
        ) | Sort-Object Name) {
            $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
            [ordered]@{
                name = $file.Name
                bytes = $file.Length
                sha256 = $hash.Hash.ToLowerInvariant()
            }
        }
    )
}

$manifestPath = Join-Path $output 'libtorch-runtime.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Output ("LibTorch runtime package: {0}" -f $output)
Write-Output ("Executable: {0}" -f $executableTarget)
Write-Output ("Native DLLs: {0}" -f $runtimeDlls.Count)
Write-Output ("Manifest: {0}" -f $manifestPath)
