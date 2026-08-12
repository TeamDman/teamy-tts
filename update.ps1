[CmdletBinding()]
param(
    # LIBTORCH is needed while compiling tch's native bindings. The runtime
    # DLLs are copied beside the installed executable below. Use the exact
    # LibTorch release paired with the tch version pinned in Cargo.toml.
    [string]$LibTorchRoot = $env:LIBTORCH,

    # This value is written to teamy-tts config.json once; it is not required
    # as an environment variable after installation.
    [string]$TorchModelDir = $env:TEAMY_TTS_TORCH_MODEL_DIR,

    # Keep a custom Cargo installation root working while making the installed
    # executable location explicit for the DLL and config bootstrap steps.
    [string]$CargoRoot = $env:CARGO_HOME
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($LibTorchRoot)) {
    throw "LibTorch is required for the tch build. Pass -LibTorchRoot <path> or set LIBTORCH to the matching tch/LibTorch release."
}

$resolvedLibTorchRoot = (Resolve-Path -LiteralPath $LibTorchRoot).Path
$libTorchLibDir = Join-Path $resolvedLibTorchRoot 'lib'
if (-not (Test-Path -LiteralPath $libTorchLibDir -PathType Container)) {
    throw "LibTorch lib directory is missing: $libTorchLibDir"
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

if ([string]::IsNullOrWhiteSpace($TorchModelDir)) {
    $knownTorchModelDir = 'G:\ml\glados-tts-upstream\models'
    if (Test-Path -LiteralPath $knownTorchModelDir -PathType Container) {
        $TorchModelDir = $knownTorchModelDir
    }
}

if (-not [string]::IsNullOrWhiteSpace($TorchModelDir)) {
    $resolvedTorchModelDir = (Resolve-Path -LiteralPath $TorchModelDir).Path
    & $installedExecutable config set --torch-model-dir $resolvedTorchModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "teamy-tts config bootstrap failed with exit code $LASTEXITCODE"
    }
}

Write-Output "Installed tch/LibTorch teamy-tts at $installedExecutable"
Write-Output "Copied $($runtimeDlls.Count) LibTorch runtime DLLs beside the executable"
if (-not [string]::IsNullOrWhiteSpace($TorchModelDir)) {
    Write-Output "Remembered TorchScript model directory: $resolvedTorchModelDir"
} else {
    Write-Output 'No TorchScript model directory was configured; use teamy-tts config set --torch-model-dir <path> when needed.'
}
