[CmdletBinding()]
param(
    # LIBTORCH is needed while compiling tch's native bindings. The runtime
    # DLLs are copied beside the installed executable below. Use the exact
    # LibTorch release paired with the tch version pinned in Cargo.toml.
    [string]$LibTorchRoot = $env:LIBTORCH,

    # Keep a custom Cargo installation root working while making the installed
    # executable location explicit for the DLL and config bootstrap steps.
    [string]$CargoRoot = $env:CARGO_HOME
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedLibTorchBuild = '2.11.0+cu128'

if ([string]::IsNullOrWhiteSpace($LibTorchRoot)) {
    throw "LibTorch is required for the tch build. Pass -LibTorchRoot <path> or set LIBTORCH to the matching tch/LibTorch release."
}

$resolvedLibTorchRoot = (Resolve-Path -LiteralPath $LibTorchRoot).Path
$libTorchLibDir = Join-Path $resolvedLibTorchRoot 'lib'
if (-not (Test-Path -LiteralPath $libTorchLibDir -PathType Container)) {
    throw "LibTorch lib directory is missing: $libTorchLibDir"
}
$buildVersionPath = Join-Path $resolvedLibTorchRoot 'build-version'
if (-not (Test-Path -LiteralPath $buildVersionPath -PathType Leaf)) {
    throw "LibTorch build-version file is missing: $buildVersionPath"
}
$actualLibTorchBuild = (Get-Content -LiteralPath $buildVersionPath -Raw).Trim()
if ($actualLibTorchBuild -ne $expectedLibTorchBuild) {
    throw "Unsupported LibTorch build '$actualLibTorchBuild'; teamy-tts is pinned to '$expectedLibTorchBuild'."
}
foreach ($requiredNativeFile in @('torch_cuda.dll', 'torch_cuda.lib')) {
    $requiredNativePath = Join-Path $libTorchLibDir $requiredNativeFile
    if (-not (Test-Path -LiteralPath $requiredNativePath -PathType Leaf)) {
        throw "Required LibTorch native file is missing: $requiredNativePath"
    }
}

# tch's build script consumes LIBTORCH while compiling its native bindings. This assignment is
# scoped to this PowerShell process and is intentionally not a persistent user
# environment-variable mutation.
$env:LIBTORCH = $resolvedLibTorchRoot

if ([string]::IsNullOrWhiteSpace($CargoRoot)) {
    $CargoRoot = Join-Path $env:USERPROFILE '.cargo'
}
$resolvedCargoRoot = [IO.Path]::GetFullPath($CargoRoot)
$installedBinDir = Join-Path $resolvedCargoRoot 'bin'
$installedExecutable = Join-Path $installedBinDir 'teamy-tts.exe'

$cargoArguments = @(
    'install'
    '--path'
    $PSScriptRoot
    '--root'
    $resolvedCargoRoot
    '--locked'
    '--force'
)
& cargo @cargoArguments
if ($LASTEXITCODE -ne 0) {
    throw "cargo install failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $installedExecutable -PathType Leaf)) {
    throw "cargo install did not produce the expected executable: $installedExecutable"
}

# cargo install does not know that the native LibTorch DLLs are part of this
# application runtime. Place them beside teamy-tts.exe so Windows can load the
# bridge without PATH being configured in every future shell.
$runtimeDlls = @(Get-ChildItem -LiteralPath $libTorchLibDir -Filter '*.dll' -File |
    Where-Object { $_.Name -ne 'torch_python.dll' } |
    Sort-Object Name)
if ($runtimeDlls.Count -eq 0) {
    throw "No LibTorch DLLs were found in $libTorchLibDir"
}
New-Item -ItemType Directory -Force -Path $installedBinDir | Out-Null
foreach ($dll in $runtimeDlls) {
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $installedBinDir $dll.Name) -Force
}

Write-Output "Installed tch/LibTorch teamy-tts at $installedExecutable"
Write-Output "Copied $($runtimeDlls.Count) LibTorch runtime DLLs beside the executable"
Write-Output 'Existing teamy-tts configuration was left unchanged; use teamy-tts config set --torch-model-dir <path> for one-time model setup.'
