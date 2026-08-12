param(
    [Parameter(Mandatory)]
    [string]$SourceDir,

    [Parameter(Mandatory)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$source = (Resolve-Path -LiteralPath $SourceDir).Path
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Native bundle source is not a directory: $source"
}

$required = @(
    'acoustic-model.bpk',
    'vocoder.bpk',
    'phonemizer.bpk',
    'frontend.tsv',
    'voice-p1.f32le',
    'voice-p2.f32le'
)
foreach ($name in $required) {
    $path = Join-Path $source $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Native bundle is missing required artifact: $path"
    }
}

$output = [IO.Path]::GetFullPath($OutputPath)
$parent = Split-Path -Parent $output
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

Compress-Archive -Path (Join-Path $source '*') -DestinationPath $output -CompressionLevel Optimal -Force
$hash = Get-FileHash -LiteralPath $output -Algorithm SHA256
$size = (Get-Item -LiteralPath $output).Length
Write-Output ("native bundle archive: {0}`nbytes: {1}`nsha256: {2}" -f $output, $size, $hash.Hash)
