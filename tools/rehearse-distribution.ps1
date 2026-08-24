[CmdletBinding()]
param(
    [string]$ExecutablePath = (Join-Path $PSScriptRoot '..\target\release\teamy-tts.exe'),
    [string]$LibTorchRoot = $env:LIBTORCH,
    [string]$NativeBundleArchive = (Join-Path $PSScriptRoot '..\artifacts\teamy-tts-glados-new-tch-native-bundle.zip'),
    [string]$RawModelArchive = (Join-Path $PSScriptRoot '..\models.zip'),
    [string]$ReceiptPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedNativeArchiveBytes = 217016604
$expectedNativeArchiveSha256 = '5fc80b76584ef7c078a417fb53e09fa8477b211e26458ad1ee8f4a25cf626e0f'
$expectedRawArchiveBytes = 343345374
$expectedRawArchiveSha256 = 'afb60dd8944934ea5c67bd85de70f424c151b5f41b50dc039578716364fa68c4'

function Resolve-RequiredFile {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Label)

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
    if (-not (Test-Path -LiteralPath $resolved.Path -PathType Leaf)) {
        throw "$Label is not a file: $($resolved.Path)"
    }
    return $resolved.Path
}

function Get-FileEvidence {
    param(
        [Parameter(Mandatory)][string]$Path,
        [string]$ExpectedSha256,
        [Nullable[long]]$ExpectedBytes
    )

    $item = Get-Item -LiteralPath $Path
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{
        path = $item.FullName
        bytes = $item.Length
        sha256 = $hash
        expected_bytes = $ExpectedBytes
        expected_sha256 = $ExpectedSha256
        verified = (($null -eq $ExpectedBytes -or $item.Length -eq $ExpectedBytes) -and
            ([string]::IsNullOrWhiteSpace($ExpectedSha256) -or $hash -eq $ExpectedSha256))
    }
}

function Get-ZipEntryNames {
    param([Parameter(Mandatory)][string]$Path)

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        return @($archive.Entries | ForEach-Object { $_.FullName } | Sort-Object)
    }
    finally {
        $archive.Dispose()
    }
}

function New-IsolatedEnvironment {
    param(
        [Parameter(Mandatory)][string]$PackageDirectory,
        [Parameter(Mandatory)][string]$HomeDirectory,
        [Parameter(Mandatory)][string]$CacheDirectory,
        [Parameter(Mandatory)][string]$TemporaryDirectory,
        [hashtable]$Overrides = @{}
    )

    $environment = @{
        # Keep only the system loader locations and the staged package. In
        # particular, do not inherit Python, Cargo, Hugging Face, or the
        # development checkout from the caller.
        PATH = "$PackageDirectory;C:\Windows\System32;C:\Windows"
        SystemRoot = $env:SystemRoot
        WINDIR = $env:WINDIR
        TEMP = $TemporaryDirectory
        TMP = $TemporaryDirectory
        TEAMY_TTS_HOME_DIR = $HomeDirectory
        TEAMY_TTS_CACHE_DIR = $CacheDirectory
        RUST_BACKTRACE = '0'
    }
    foreach ($entry in $Overrides.GetEnumerator()) {
        if ($null -eq $entry.Value) {
            $environment.Remove($entry.Key)
        }
        else {
            $environment[$entry.Key] = [string]$entry.Value
        }
    }
    return $environment
}

$script:CommandRecords = [System.Collections.Generic.List[object]]::new()
$script:CommandNumber = 0

function Invoke-IsolatedCommand {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][hashtable]$Environment,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$LogDirectory,
        [string]$InputText
    )

    $script:CommandNumber++
    $name = '{0:D2}-{1}' -f $script:CommandNumber, ($Label -replace '[^A-Za-z0-9._-]', '-')
    $stdoutPath = Join-Path $LogDirectory "$name.stdout.txt"
    $stderrPath = Join-Path $LogDirectory "$name.stderr.txt"
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        [void]$startInfo.ArgumentList.Add($argument)
    }

    # ProcessStartInfo inherits the caller's environment by default. Clear it
    # before applying the intentionally small rehearsal environment.
    foreach ($key in @($startInfo.Environment.Keys)) {
        [void]$startInfo.Environment.Remove($key)
    }
    foreach ($entry in $Environment.GetEnumerator()) {
        if ($null -ne $entry.Value) {
            $startInfo.Environment[$entry.Key] = [string]$entry.Value
        }
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $started = [DateTimeOffset]::UtcNow
    $exitCode = -1
    $stdout = ''
    $stderr = ''
    try {
        if (-not $process.Start()) {
            throw "ProcessStart returned false for $Executable"
        }
        if ($null -ne $InputText) {
            $process.StandardInput.Write($InputText)
        }
        $process.StandardInput.Close()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
    }
    catch {
        $stderr = $_.Exception.ToString()
    }
    finally {
        $process.Dispose()
    }

    [IO.File]::WriteAllText($stdoutPath, $stdout, [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($stderrPath, $stderr, [Text.UTF8Encoding]::new($false))
    $ended = [DateTimeOffset]::UtcNow
    $record = [ordered]@{
        label = $Label
        executable = $Executable
        arguments = $Arguments
        exit_code = $exitCode
        started_at_utc = $started.ToString('O')
        ended_at_utc = $ended.ToString('O')
        elapsed_ms = ($ended - $started).TotalMilliseconds
        stdout_path = $stdoutPath
        stdout_bytes = ([Text.Encoding]::UTF8.GetByteCount($stdout))
        stdout_sha256 = (Get-FileHash -LiteralPath $stdoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
        stderr_path = $stderrPath
        stderr_bytes = ([Text.Encoding]::UTF8.GetByteCount($stderr))
        stderr_sha256 = (Get-FileHash -LiteralPath $stderrPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $script:CommandRecords.Add($record)
    return [pscustomobject]@{ Record = $record; Stdout = $stdout; Stderr = $stderr }
}

function Assert-Success {
    param([Parameter(Mandatory)]$Result, [Parameter(Mandatory)][string]$Label)
    if ($Result.Record.exit_code -ne 0) {
        throw "$Label failed with exit code $($Result.Record.exit_code); see $($Result.Record.stderr_path)"
    }
}

function Assert-Failure {
    param([Parameter(Mandatory)]$Result, [Parameter(Mandatory)][string]$Label)
    if ($Result.Record.exit_code -eq 0) {
        throw "$Label unexpectedly succeeded"
    }
}

function Read-JsonStdout {
    param([Parameter(Mandatory)]$Result, [Parameter(Mandatory)][string]$Label)
    Assert-Success $Result $Label
    if ([string]::IsNullOrWhiteSpace($Result.Stdout)) {
        throw "$Label produced no JSON on stdout"
    }
    try {
        return $Result.Stdout | ConvertFrom-Json
    }
    catch {
        throw "$Label produced invalid JSON: $($_.Exception.Message); see $($Result.Record.stdout_path)"
    }
}

function Add-Assertion {
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][System.Collections.Generic.List[object]]$Assertions,
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][bool]$Passed,
        [Parameter(Mandatory)][string]$Evidence
    )
    $Assertions.Add([ordered]@{ id = $Id; passed = $Passed; evidence = $Evidence })
    if (-not $Passed) {
        throw "rehearsal assertion failed: $Id ($Evidence)"
    }
}

$startedAt = [DateTimeOffset]::UtcNow
$assertions = [System.Collections.Generic.List[object]]::new()
$receipt = [ordered]@{
    schema_version = 1
    kind = 'teamy-tts-distribution-rehearsal'
    status = 'failed'
    started_at_utc = $startedAt.ToString('O')
    completed_at_utc = $null
    scope = 'local installed-executable rehearsal; no Cloudflare, Terraform, DNS, credential, or remote publication mutation'
    repository = $null
    inputs = $null
    package = $null
    commands = $script:CommandRecords
    assertions = $assertions
    environment_policy = [ordered]@{
        inherited_environment_cleared = $true
        path = 'staged-package;C:\Windows\System32;C:\Windows'
        python_required = $false
        hugging_face_credentials_used = $false
        development_checkout_visible_to_child_process = $false
    }
    external_actions = [ordered]@{
        cloudflare_contacted = $false
        terraform_applied = $false
        dns_changed = $false
        credentials_loaded = $false
        remote_model_downloaded = $false
    }
}

$stageRoot = $null
try {
    $executable = Resolve-RequiredFile $ExecutablePath 'executable'
    $libtorch = Resolve-Path -LiteralPath $LibTorchRoot -ErrorAction Stop
    $libtorch = $libtorch.Path
    $nativeArchive = Resolve-RequiredFile $NativeBundleArchive 'native bundle archive'
    $rawArchive = Resolve-RequiredFile $RawModelArchive 'raw model archive'

    $gitRevision = (& git -C (Split-Path -Parent $PSScriptRoot) rev-parse HEAD).Trim()
    $gitStatus = @(& git -C (Split-Path -Parent $PSScriptRoot) status --short)
    $receipt.repository = [ordered]@{
        path = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
        head = $gitRevision
        dirty = ($gitStatus.Count -gt 0)
        dirty_paths = @($gitStatus | ForEach-Object { $_.Substring(3).Trim() })
    }

    $nativeEvidence = Get-FileEvidence $nativeArchive $expectedNativeArchiveSha256 $expectedNativeArchiveBytes
    $rawEvidence = Get-FileEvidence $rawArchive $expectedRawArchiveSha256 $expectedRawArchiveBytes
    Add-Assertion $assertions 'publication.native-archive-hash' $nativeEvidence.verified ("bytes={0}, sha256={1}" -f $nativeEvidence.bytes, $nativeEvidence.sha256)
    Add-Assertion $assertions 'publication.raw-archive-hash' $rawEvidence.verified ("bytes={0}, sha256={1}" -f $rawEvidence.bytes, $rawEvidence.sha256)
    $receipt.inputs = [ordered]@{
        native_bundle = $nativeEvidence
        native_bundle_entries = @(Get-ZipEntryNames $nativeArchive)
        raw_model_archive = $rawEvidence
        libtorch_root = $libtorch
        libtorch_build = (Get-Content -LiteralPath (Join-Path $libtorch 'build-version') -Raw).Trim()
    }
    Add-Assertion $assertions 'toolchain.libtorch-build' ($receipt.inputs.libtorch_build -eq '2.11.0+cu128') ("build={0}" -f $receipt.inputs.libtorch_build)

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmssfff'
    $stageRoot = Join-Path $PSScriptRoot "..\target\distribution-rehearsal\$stamp"
    $stageRoot = [IO.Path]::GetFullPath($stageRoot)
    $packageDirectory = Join-Path $stageRoot 'package'
    $inputDirectory = Join-Path $packageDirectory 'publication-input'
    $logDirectory = Join-Path $stageRoot 'logs'
    $homeDirectory = Join-Path $stageRoot 'app-home'
    $cacheDirectory = Join-Path $stageRoot 'cache'
    $workDirectory = Join-Path $stageRoot 'work'
    New-Item -ItemType Directory -Force -Path $inputDirectory,$logDirectory,$homeDirectory,$cacheDirectory,$workDirectory | Out-Null

    & (Join-Path $PSScriptRoot 'package-libtorch-runtime.ps1') `
        -ExecutablePath $executable `
        -LibTorchRoot $libtorch `
        -OutputDir $packageDirectory | Out-Host
    $runtimeManifestPath = Join-Path $packageDirectory 'libtorch-runtime.json'
    $runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw | ConvertFrom-Json
    $packageFiles = @(Get-ChildItem -LiteralPath $packageDirectory -File -Recurse)
    Add-Assertion $assertions 'package.no-python-runtime' (-not ($packageFiles.Name -contains 'torch_python.dll') -and -not ($packageFiles.Extension -contains '.py')) 'torch_python.dll and Python source files absent'
    Add-Assertion $assertions 'package.runtime-manifest' ($runtimeManifest.python_required -eq $false -and $runtimeManifest.native_files.Count -gt 0) ("native-files={0}, python-required={1}" -f $runtimeManifest.native_files.Count, $runtimeManifest.python_required)

    $stagedArchive = Join-Path $inputDirectory 'teamy-tts-glados-new-tch-native-bundle.zip'
    Copy-Item -LiteralPath $nativeArchive -Destination $stagedArchive
    $stagedEvidence = Get-FileEvidence $stagedArchive $expectedNativeArchiveSha256 $expectedNativeArchiveBytes
    Add-Assertion $assertions 'package.staged-native-archive' $stagedEvidence.verified ("path={0}" -f $stagedArchive)

    $receipt.package = [ordered]@{
        root = $packageDirectory
        executable = (Get-FileEvidence (Join-Path $packageDirectory 'teamy-tts.exe'))
        runtime_manifest = $runtimeManifest
        staged_native_archive = $stagedEvidence
        staged_files = @($packageFiles | ForEach-Object { $_.Name } | Sort-Object)
    }

    $baseEnvironment = New-IsolatedEnvironment $packageDirectory $homeDirectory $cacheDirectory $env:TEMP
    $versionResult = Invoke-IsolatedCommand 'version' (Join-Path $packageDirectory 'teamy-tts.exe') @('--version') $baseEnvironment $workDirectory $logDirectory
    Assert-Success $versionResult 'staged --version'
    Add-Assertion $assertions 'package.version-reports-revision' ($versionResult.Stdout -match '0\.1\.0') $versionResult.Stdout.Trim()

    $prepareResult = Invoke-IsolatedCommand 'prepare-native-bundle' (Join-Path $packageDirectory 'teamy-tts.exe') @('--output-format','json','model','prepare','glados','--source-archive',$stagedArchive,'--force') $baseEnvironment $workDirectory $logDirectory
    $prepareReport = Read-JsonStdout $prepareResult 'model prepare'
    $preparedDirectory = [IO.Path]::GetFullPath([string]$prepareReport.prepared_dir)
    Add-Assertion $assertions 'model.prepared-directory' (Test-Path -LiteralPath $preparedDirectory -PathType Container) $preparedDirectory

    $configResult = Invoke-IsolatedCommand 'remember-torch-model-dir' (Join-Path $packageDirectory 'teamy-tts.exe') @('--output-format','json','config','set','--torch-model-dir',$preparedDirectory,'--torch-device','0') $baseEnvironment $workDirectory $logDirectory
    $configReport = Read-JsonStdout $configResult 'config set'
    Add-Assertion $assertions 'configuration.remembered-model-dir' ([string]$configReport.effective.torch_model_dir -eq $preparedDirectory) ([string]$configReport.effective.torch_model_dir)

    $doctorEnvironment = New-IsolatedEnvironment $packageDirectory $homeDirectory $cacheDirectory $env:TEMP
    $doctorResult = Invoke-IsolatedCommand 'doctor-deep-remembered-config' (Join-Path $packageDirectory 'teamy-tts.exe') @('--output-format','json','doctor','--deep','--offline') $doctorEnvironment $workDirectory $logDirectory
    $doctorReport = Read-JsonStdout $doctorResult 'deep doctor with remembered config'
    Add-Assertion $assertions 'doctor.typed-report' ($doctorReport.'schema-version' -eq 1 -and $null -ne $doctorReport.checks) ("schema={0}, checks={1}" -f $doctorReport.'schema-version', $doctorReport.checks.Count)
    Add-Assertion $assertions 'doctor.deep-pass' ($doctorReport.status -eq 'pass') ("status={0}" -f $doctorReport.status)
    Add-Assertion $assertions 'doctor.remembered-provenance' (($doctorReport.checks | Where-Object { $_.id -eq 'configuration.precedence' }).evidence -match 'remembered-config') (($doctorReport.checks | Where-Object { $_.id -eq 'configuration.precedence' }).evidence)

    $overrideEnvironment = New-IsolatedEnvironment $packageDirectory $homeDirectory $cacheDirectory $env:TEMP @{ TEAMY_TTS_TORCH_MODEL_DIR = $preparedDirectory }
    $overrideDoctorResult = Invoke-IsolatedCommand 'doctor-environment-override' (Join-Path $packageDirectory 'teamy-tts.exe') @('--output-format','json','doctor','--offline') $overrideEnvironment $workDirectory $logDirectory
    $overrideDoctor = Read-JsonStdout $overrideDoctorResult 'doctor with environment override'
    Add-Assertion $assertions 'doctor.environment-provenance' (($overrideDoctor.checks | Where-Object { $_.id -eq 'configuration.precedence' }).evidence -match 'environment-override:TEAMY_TTS_TORCH_MODEL_DIR') (($overrideDoctor.checks | Where-Object { $_.id -eq 'configuration.precedence' }).evidence)

    $benchmarkResult = Invoke-IsolatedCommand 'benchmark-cold-and-warm' (Join-Path $packageDirectory 'teamy-tts.exe') @('--output-format','json','benchmark','Hello, friend','--warmups','1','--measurements','3') $doctorEnvironment $workDirectory $logDirectory
    $benchmarkReport = Read-JsonStdout $benchmarkResult 'benchmark'
    Add-Assertion $assertions 'benchmark.correctness' ($benchmarkReport.correctness_passed -eq $true) ("median-ms={0}, sample-count={1}" -f $benchmarkReport.median_ms, $benchmarkReport.sample_count)
    Add-Assertion $assertions 'benchmark.warm-resident-measurement' ($benchmarkReport.measurement_count -eq 3 -and $benchmarkReport.model_load_ms -gt 0) ("load-ms={0}, measurements={1}" -f $benchmarkReport.model_load_ms, $benchmarkReport.measurement_count)

    $writePath = Join-Path $stageRoot 'audio\hello.wav'
    $writeResult = Invoke-IsolatedCommand 'write-explicit-wav' (Join-Path $packageDirectory 'teamy-tts.exe') @('write','Hello, friend','--output',$writePath) $doctorEnvironment $workDirectory $logDirectory
    Assert-Success $writeResult 'write with explicit output'
    $writeEvidence = Get-FileEvidence $writePath
    Add-Assertion $assertions 'write.opt-in-wav' ($writeEvidence.bytes -gt 44 -and $writeResult.Stdout.Trim() -eq $writePath) ("path={0}, bytes={1}" -f $writeResult.Stdout.Trim(), $writeEvidence.bytes)

    $sayResult = Invoke-IsolatedCommand 'say-in-memory-volume-zero' (Join-Path $packageDirectory 'teamy-tts.exe') @('say','Hello, friend','--volume','0') $doctorEnvironment $workDirectory $logDirectory
    Assert-Success $sayResult 'say in-memory playback'
    Add-Assertion $assertions 'say.volume-zero' ($sayResult.Stderr -match 'volume.{0,20}0') ('volume-zero-log={0}' -f ($sayResult.Stderr -match 'volume.{0,20}0'))
    Add-Assertion $assertions 'say.no-persistent-output' ([string]::IsNullOrWhiteSpace($sayResult.Stdout) -and -not (Test-Path -LiteralPath (Join-Path $workDirectory 'outputs'))) ('stdout-bytes={0}, outputs-directory={1}' -f $sayResult.Record.stdout_bytes, (Test-Path (Join-Path $workDirectory 'outputs')))

    $interactiveResult = Invoke-IsolatedCommand 'interactive-resident-two-lines-volume-zero' (Join-Path $packageDirectory 'teamy-tts.exe') @('interactive','--volume','0') $doctorEnvironment $workDirectory $logDirectory "Hello, friend`r`nThe letter eɪ`r`n"
    Assert-Success $interactiveResult 'interactive resident playback'
    Add-Assertion $assertions 'interactive.volume-zero' ($interactiveResult.Stderr -match 'volume.{0,20}0') ('volume-zero-log={0}' -f ($interactiveResult.Stderr -match 'volume.{0,20}0'))
    Add-Assertion $assertions 'interactive.resident-no-output' ([string]::IsNullOrWhiteSpace($interactiveResult.Stdout) -and -not (Test-Path -LiteralPath (Join-Path $workDirectory 'outputs'))) ('stdout-bytes={0}' -f $interactiveResult.Record.stdout_bytes)

    $missingCache = Join-Path $stageRoot 'missing-cache'
    $missingEnvironment = New-IsolatedEnvironment $packageDirectory $homeDirectory $missingCache $env:TEMP
    $missingResult = Invoke-IsolatedCommand 'missing-prepared-model' (Join-Path $packageDirectory 'teamy-tts.exe') @('say','Hello, friend') $missingEnvironment $workDirectory $logDirectory
    Assert-Failure $missingResult 'say with missing prepared model'
    Add-Assertion $assertions 'failure.missing-model' $true ("exit-code={0}" -f $missingResult.Record.exit_code)

    $modelRoot = Split-Path -Parent (Split-Path -Parent $preparedDirectory)
    $corruptModelRoot = Join-Path $stageRoot 'corrupt-models'
    New-Item -ItemType Directory -Force -Path $corruptModelRoot | Out-Null
    Get-ChildItem -LiteralPath $modelRoot -Force | Copy-Item -Destination $corruptModelRoot -Recurse -Force
    $corruptPreparedDirectory = Join-Path $corruptModelRoot (Split-Path -Leaf (Split-Path -Parent $preparedDirectory)) (Split-Path -Leaf $preparedDirectory)
    [IO.File]::WriteAllText((Join-Path $corruptPreparedDirectory 'manifest.json'), '{"corrupt":true}', [Text.UTF8Encoding]::new($false))
    $corruptEnvironment = New-IsolatedEnvironment $packageDirectory $homeDirectory (Join-Path $stageRoot 'corrupt-cache') $env:TEMP @{
        TEAMY_TTS_MODEL_DIR = $corruptModelRoot
        TEAMY_TTS_TORCH_MODEL_DIR = $corruptPreparedDirectory
    }
    $corruptDoctorResult = Invoke-IsolatedCommand 'corrupt-model-doctor' (Join-Path $packageDirectory 'teamy-tts.exe') @('--output-format','json','doctor','--deep','--offline') $corruptEnvironment $workDirectory $logDirectory
    $corruptDoctor = Read-JsonStdout $corruptDoctorResult 'doctor with corrupt model'
    Add-Assertion $assertions 'failure.corrupt-model-report' ($corruptDoctor.status -eq 'fail') ("status={0}" -f $corruptDoctor.status)
    $corruptSayResult = Invoke-IsolatedCommand 'corrupt-model-say' (Join-Path $packageDirectory 'teamy-tts.exe') @('say','Hello, friend') $corruptEnvironment $workDirectory $logDirectory
    Assert-Failure $corruptSayResult 'say with corrupt model'
    Add-Assertion $assertions 'failure.corrupt-model-say' $true ("exit-code={0}" -f $corruptSayResult.Record.exit_code)

    $brokenRuntimeDirectory = Join-Path $stageRoot 'broken-runtime'
    Copy-Item -LiteralPath $packageDirectory -Destination $brokenRuntimeDirectory -Recurse
    Remove-Item -LiteralPath (Join-Path $brokenRuntimeDirectory 'torch_cuda.dll') -Force
    $brokenRuntimeEnvironment = New-IsolatedEnvironment $brokenRuntimeDirectory $homeDirectory $cacheDirectory $env:TEMP
    $brokenRuntimeResult = Invoke-IsolatedCommand 'missing-native-runtime' (Join-Path $brokenRuntimeDirectory 'teamy-tts.exe') @('--version') $brokenRuntimeEnvironment $workDirectory $logDirectory
    Assert-Failure $brokenRuntimeResult 'version with missing native runtime'
    Add-Assertion $assertions 'failure.missing-native-runtime' $true ("exit-code={0}" -f $brokenRuntimeResult.Record.exit_code)

    $receipt.status = 'pass'
}
catch {
    $receipt.failure = $_.Exception.ToString()
    throw
}
finally {
    $completedAt = [DateTimeOffset]::UtcNow
    $receipt.completed_at_utc = $completedAt.ToString('O')
    if ([string]::IsNullOrWhiteSpace($ReceiptPath)) {
        if ($null -ne $stageRoot) {
            $ReceiptPath = Join-Path $stageRoot 'receipt.json'
        }
        else {
            $ReceiptPath = Join-Path $PSScriptRoot '..\target\distribution-rehearsal\receipt.json'
        }
    }
    $receiptParent = Split-Path -Parent ([IO.Path]::GetFullPath($ReceiptPath))
    New-Item -ItemType Directory -Force -Path $receiptParent | Out-Null
    $receipt | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8NoBOM
    Write-Output "Distribution rehearsal receipt: $([IO.Path]::GetFullPath($ReceiptPath))"
}
