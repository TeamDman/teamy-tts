<#
.SYNOPSIS
Loads the Cloudflare and R2 Terraform environment into the current
PowerShell session from the fixed 1Password item.

Use this file by dot-sourcing it:

    . .\get-cloudflare-token.ps1

Running it with the call operator (&) would set the variables only in the
child PowerShell process, so Terraform would not see them afterward.
#>

$itemReference = "op://Private/cloudflare teamy-tts-terraform"
$fallbackOpPath = "C:\Users\Teamy\AppData\Local\Microsoft\WinGet\Packages\AgileBits.1Password.CLI_Microsoft.Winget.Source_8wekyb3d8bbwe\.\op.exe"

$opCommand = Get-Command op -ErrorAction SilentlyContinue
if ($null -ne $opCommand) {
    $opExecutable = $opCommand.Source
} elseif (Test-Path -LiteralPath $fallbackOpPath) {
    $opExecutable = $fallbackOpPath
} else {
    throw "The 1Password CLI ('op') is not available on PATH or at its configured WinGet path. Install it and authenticate before loading the Cloudflare environment."
}

function Read-OpField {
    param(
        [Parameter(Mandatory)]
        [string]$Field
    )

    $output = & $opExecutable read "$itemReference/$Field"
    if ($LASTEXITCODE -ne 0) {
        throw "1Password could not read $Field from $itemReference."
    }

    $value = ($output -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "1Password returned an empty value for $Field."
    }

    return $value
}

$env:CLOUDFLARE_API_TOKEN = Read-OpField "credential"
$env:AWS_ACCESS_KEY_ID = Read-OpField "access key id"
$env:AWS_SECRET_ACCESS_KEY = Read-OpField "secret access key"

Write-Host "Cloudflare and R2 Terraform credentials loaded from 1Password for this PowerShell session."
