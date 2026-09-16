[CmdletBinding()]
param(
	[ValidateSet('cuda-native', 'tch-native')]
	[string]$Backend = 'cuda-native'
)

function Invoke-Step {
	param(
		[Parameter(Mandatory = $true)]
		[string]$Label,
		[Parameter(Mandatory = $true)]
		[scriptblock]$Action
	)

	Write-Host -ForegroundColor Yellow "Running $Label..."
	& $Action
	if ($LASTEXITCODE -ne 0) {
		throw "$Label failed with exit code $LASTEXITCODE"
	}
}

function Get-NonTracyTestFeatureArgs {
	$metadata = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
	$pkg = if ($metadata.packages.Count -eq 1) {
		$metadata.packages[0]
	} else {
		$manifestPath = (Resolve-Path (Join-Path (Get-Location) 'Cargo.toml')).Path
		$metadata.packages |
			Where-Object { $_.manifest_path -eq $manifestPath } |
			Select-Object -First 1
	}
	if (-not $pkg) {
		throw "Could not determine root package from cargo metadata"
	}

	$features = @($Backend) + @($pkg.features.PSObject.Properties.Name | Where-Object { $_ -notin @("default", "tracy", "cuda-native", "tch-native") })
	if ($features.Count -gt 0) {
		return @("--no-default-features", "--features", ($features -join ","))
	}

	return @()
}

Invoke-Step -Label "format check" -Action {
	rustup run nightly -- cargo fmt --all
}

Invoke-Step -Label "clippy lint check" -Action {
	cargo clippy --release --no-default-features --features "$Backend,tracy" -- -D warnings
}

Invoke-Step -Label "build" -Action {
	cargo build --release --no-default-features --features "$Backend,tracy" --quiet
}

Invoke-Step -Label "tests" -Action {
	$featuresArg = Get-NonTracyTestFeatureArgs
	cargo test --release @featuresArg --quiet
}
