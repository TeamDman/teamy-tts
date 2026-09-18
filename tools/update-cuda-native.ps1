[CmdletBinding()]
param(
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$CudnnRoot = $env:CUDNN_ROOT,
    [string]$NativeModelDir = $env:TEAMY_TTS_NATIVE_MODEL_DIR,
    [string]$CargoRoot = $env:CARGO_HOME
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
if ([string]::IsNullOrWhiteSpace($CargoRoot)) {
    $CargoRoot = Join-Path $env:USERPROFILE '.cargo'
}
$installRoot = [IO.Path]::GetFullPath($CargoRoot)
$binDir = Join-Path $installRoot 'bin'
$executable = Join-Path $binDir 'teamy-tts.exe'
$modelTarget = Join-Path $installRoot 'share/teamy-tts/glados-native-v1'

if ([string]::IsNullOrWhiteSpace($CudaRoot)) {
    throw 'Set CUDA_PATH or pass -CudaRoot to the CUDA toolkit used for this build.'
}
$cuda = (Resolve-Path -LiteralPath $CudaRoot).Path
$cudaDirs = @((Join-Path $cuda 'bin/x64'), (Join-Path $cuda 'bin'))
$cudaBin = $cudaDirs | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'cublas64_13.dll') } | Select-Object -First 1
if (-not $cudaBin) {
    # Other supported toolkit majors use the same directory layouts.
    $cudaBin = $cudaDirs | Where-Object {
        (Test-Path -LiteralPath $_ -PathType Container) -and @(Get-ChildItem -LiteralPath $_ -Filter 'cublas64_*.dll' -File).Count -gt 0
    } | Select-Object -First 1
}
if (-not $cudaBin) { throw "CUDA runtime DLLs were not found beneath $cuda" }

if ([string]::IsNullOrWhiteSpace($CudnnRoot)) {
    if ($env:GLADOS_CUDNN_LIBRARY) {
        $CudnnRoot = Split-Path ([IO.Path]::GetFullPath($env:GLADOS_CUDNN_LIBRARY)) -Parent
    } else {
        $CudnnRoot = $binDir
    }
}
$cudnn = (Resolve-Path -LiteralPath $CudnnRoot).Path
$cudnnDirs = @($cudnn, (Join-Path $cudnn 'bin'), (Join-Path $cudnn 'lib'), (Join-Path $cudnn 'bin/x64'))
$cudnnBin = $cudnnDirs | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'cudnn64_9.dll') -PathType Leaf } | Select-Object -First 1
if (-not $cudnnBin) { throw 'Pass -CudnnRoot with the cuDNN 9 runtime DLLs and their CUDA dependencies.' }
foreach ($name in @('cudnn64_9.dll', 'cudnn_adv64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_graph64_9.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $cudnnBin $name) -PathType Leaf)) {
        throw "Required cuDNN runtime is missing: $name"
    }
}

if ([string]::IsNullOrWhiteSpace($NativeModelDir)) {
    foreach ($candidate in @($modelTarget, (Join-Path $repoRoot 'artifacts/native-glados'))) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'weights.safetensors') -PathType Leaf) {
            $NativeModelDir = $candidate
            break
        }
    }
}
$modelSource = $null
if (-not [string]::IsNullOrWhiteSpace($NativeModelDir)) {
    $modelSource = (Resolve-Path -LiteralPath $NativeModelDir).Path
    foreach ($name in @('weights.safetensors', 'frontend.tsv')) {
        if (-not (Test-Path -LiteralPath (Join-Path $modelSource $name) -PathType Leaf)) {
            throw "Required native model artifact is missing: $name"
        }
    }
}

# A CUDA-12 cuDNN distribution may need cublasLt64_12 even when our custom
# kernels link CUDA 13. Keep each distribution's vendor dependencies together.
# Deliberately exclude torch/c10/Python DLLs.
$runtimeFiles = @{}
foreach ($source in @(
    @{ Directory = $cudnnBin; Patterns = @('cudnn*64_9.dll', 'cublas*.dll', 'cudart*.dll', 'nvrtc*.dll', 'nvJitLink*.dll', 'zlibwapi.dll') },
    @{ Directory = $cudaBin; Patterns = @('cublas*.dll', 'cudart*.dll') }
)) {
    foreach ($pattern in $source.Patterns) {
        foreach ($file in Get-ChildItem -LiteralPath $source.Directory -Filter $pattern -File) {
            $runtimeFiles[$file.Name] = $file.FullName
        }
    }
}

# Only this process needs the compiler settings; installed commands use their
# adjacent DLLs and durable config, without changes to the user's environment.
$savedCudaPath = $env:CUDA_PATH
$savedPath = $env:PATH
try {
    $env:CUDA_PATH = $cuda
    $env:PATH = "$(Join-Path $cuda 'bin');$cudaBin;$savedPath"
    & cargo install --path $repoRoot --root $installRoot --locked --force
    if ($LASTEXITCODE -ne 0) { throw "cargo install failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Missing installed executable: $executable" }

    foreach ($entry in $runtimeFiles.GetEnumerator()) {
        $destination = Join-Path $binDir $entry.Key
        if (-not [string]::Equals($entry.Value, $destination, [StringComparison]::OrdinalIgnoreCase)) {
            Copy-Item -LiteralPath $entry.Value -Destination $destination -Force
        }
    }
    if ($null -eq $modelSource) {
        $acquiredJson = & $executable --output-format json model acquire-prepared Teamy
        if ($LASTEXITCODE -ne 0) { throw 'Failed to acquire the verified safetensors model bundle.' }
        $acquired = ($acquiredJson -join "`n") | ConvertFrom-Json
        $modelSource = $acquired.prepared_dir
    }
    $null = New-Item -ItemType Directory -Force -Path $modelTarget
    foreach ($name in @('weights.safetensors', 'frontend.tsv', 'manifest.json')) {
        $source = Join-Path $modelSource $name
        $destination = Join-Path $modelTarget $name
        if ((Test-Path -LiteralPath $source -PathType Leaf) -and
            -not [string]::Equals($source, $destination, [StringComparison]::OrdinalIgnoreCase)) {
            Copy-Item -LiteralPath $source -Destination $destination -Force
        }
    }
    & $executable config set --backend cuda-native --native-model-dir $modelTarget
    if ($LASTEXITCODE -ne 0) { throw 'Failed to configure the installed native backend.' }

    & cargo build --manifest-path (Join-Path $repoRoot 'sapi/Cargo.toml') --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Failed to build the SAPI adapter.' }
    $sapiBuildRoot = if ($env:CARGO_TARGET_DIR) { [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR) } else { Join-Path $repoRoot 'sapi/target' }
    $sapiDll = Join-Path $sapiBuildRoot 'release/teamy_tts_sapi.dll'
    $sapiHash = (Get-FileHash -LiteralPath $sapiDll -Algorithm SHA256).Hash.ToLowerInvariant()
    $sapiDir = Join-Path $installRoot "share/teamy-tts/sapi/$sapiHash"
    $null = New-Item -ItemType Directory -Force -Path $sapiDir
    $sapiTarget = Join-Path $sapiDir 'teamy_tts_sapi.dll'
    if (-not (Test-Path -LiteralPath $sapiTarget)) {
        Copy-Item -LiteralPath $sapiDll -Destination $sapiTarget
    } elseif ((Get-FileHash -LiteralPath $sapiTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sapiHash) {
        throw 'Installed versioned SAPI DLL failed its hash check.'
    }
    [IO.File]::WriteAllText((Join-Path $installRoot 'share/teamy-tts/sapi/current.txt'), $sapiTarget)

    # Check the installed loader with only Windows system directories on PATH.
    # Temporarily clear inference overrides so the durable install is what runs.
    $overrideNames = @('GLADOS_CUDNN_LIBRARY', 'TEAMY_TTS_BACKEND', 'TEAMY_TTS_NATIVE_MODEL_DIR')
    $savedOverrides = @{}
    foreach ($name in $overrideNames) {
        $savedOverrides[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        Remove-Item -LiteralPath "Env:\$name" -ErrorAction SilentlyContinue
    }
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        $doctorJson = & $executable --output-format json doctor --offline --deep
        if ($LASTEXITCODE -ne 0) { throw 'Installed native doctor failed to execute.' }
        $doctor = ($doctorJson -join "`n") | ConvertFrom-Json
        if ($doctor.status -ne 'pass') { throw "Installed native doctor reported $($doctor.status): $doctorJson" }
    } finally {
        foreach ($name in $overrideNames) {
            if ($null -eq $savedOverrides[$name]) {
                Remove-Item -LiteralPath "Env:\$name" -ErrorAction SilentlyContinue
            } else {
                [Environment]::SetEnvironmentVariable($name, $savedOverrides[$name], 'Process')
            }
        }
    }
    Write-Output "Installed CUDA-native teamy-tts at $executable"
    Write-Output "Native model: $modelTarget"
    Write-Output "Runtime DLLs: $($runtimeFiles.Count); isolated deep doctor passed."
    Write-Output "SAPI adapter: $sapiTarget"
    Write-Output 'To register or update the selectable Windows voice, run teamy-tts sapi install from an administrator terminal. This preserves the system default voice.'
} finally {
    $env:CUDA_PATH = $savedCudaPath
    $env:PATH = $savedPath
}
