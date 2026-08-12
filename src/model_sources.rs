use crate::model_registry::ModelDefinition;
use crate::model_registry::{self};
use eyre::Context;
use eyre::Result;
use eyre::bail;
use eyre::eyre;
use facet::Facet;
use futures_util::StreamExt;
use reqwest::Client;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;
use teamy_cancellation::CancellationToken;
use tokio::io::AsyncWriteExt;

pub const TEAMY_SOURCE_URL_ENV_VAR: &str = "TEAMY_TTS_TEAMY_SOURCE_URL";
pub const R2D2FISH_ONEDRIVE_URL_ENV_VAR: &str = "TEAMY_TTS_R2D2FISH_ONEDRIVE_URL";
pub const TEAMY_NATIVE_SOURCE_URL_ENV_VAR: &str = "TEAMY_TTS_TEAMY_NATIVE_SOURCE_URL";
pub const R2D2FISH_ONEDRIVE_NATIVE_SOURCE_URL_ENV_VAR: &str =
    "TEAMY_TTS_R2D2FISH_ONEDRIVE_NATIVE_SOURCE_URL";

const TEAMY_RAW_SOURCE_URL: &str = "https://pub-efc9d45264d54fffb27e33d408633ea8.r2.dev/raw/glados/afb60dd8944934ea5c67bd85de70f424c151b5f41b50dc039578716364fa68c4/models.zip";
const TEAMY_NATIVE_SOURCE_URL: &str = "https://pub-efc9d45264d54fffb27e33d408633ea8.r2.dev/native/glados/ab663a68fb5263b8df49f76b80812ba2692b5d1a0234a246528d65d89fd2f81f/native-bundle.zip";

#[derive(Clone, Copy, Debug)]
struct SourceDefinition {
    id: &'static str,
    display_name: &'static str,
    raw_url_env_var: &'static str,
    native_url_env_var: &'static str,
    raw_url: Option<&'static str>,
    native_url: Option<&'static str>,
}

const SOURCES: &[SourceDefinition] = &[
    SourceDefinition {
        id: "Teamy",
        display_name: "Teamy Cloudflare R2",
        raw_url_env_var: TEAMY_SOURCE_URL_ENV_VAR,
        native_url_env_var: TEAMY_NATIVE_SOURCE_URL_ENV_VAR,
        raw_url: Some(TEAMY_RAW_SOURCE_URL),
        native_url: Some(TEAMY_NATIVE_SOURCE_URL),
    },
    SourceDefinition {
        id: "R2D2FISH-OneDrive",
        display_name: "R2D2FISH OneDrive",
        raw_url_env_var: R2D2FISH_ONEDRIVE_URL_ENV_VAR,
        native_url_env_var: R2D2FISH_ONEDRIVE_NATIVE_SOURCE_URL_ENV_VAR,
        raw_url: None,
        native_url: None,
    },
];

/// The durable receipt emitted after an archive has passed verification.
#[derive(Debug, Facet)]
pub struct AcquisitionReport {
    pub model: String,
    pub source: String,
    pub source_display_name: String,
    pub source_url: String,
    pub archive_path: String,
    pub acquisition_receipt_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub verified: bool,
}

/// The durable receipt emitted after a native bundle archive has passed
/// verification.
#[derive(Debug, Facet)]
pub struct NativeAcquisitionReport {
    pub model: String,
    pub source: String,
    pub source_display_name: String,
    pub source_url: String,
    pub archive_path: String,
    pub acquisition_receipt_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub verified: bool,
}

/// Download, verify, and atomically install a raw model archive.
///
/// # Errors
///
/// Returns an error when the source is unknown or unconfigured, the HTTP
/// request fails, cancellation is requested, or archive verification fails.
pub async fn acquire(
    model: ModelDefinition,
    source_selector: &str,
    cancellation_token: CancellationToken,
) -> Result<AcquisitionReport> {
    let source = resolve_source(source_selector)?;
    let source_url = resolve_source_url(
        source.id,
        source.raw_url_env_var,
        source.raw_url,
        "raw model archive",
    )?;

    let archive_path = model_registry::raw_archive_path(model);
    let receipt_path = model_registry::acquisition_receipt_path(model);
    let parent = archive_path
        .parent()
        .ok_or_else(|| eyre!("archive path has no parent: {}", archive_path.display()))?;
    tokio::fs::create_dir_all(parent)
        .await
        .wrap_err_with(|| format!("failed to create raw model directory {}", parent.display()))?;

    let partial_path = parent.join("models.zip.partial");
    let client = Client::new();
    let response = client
        .get(&source_url)
        .send()
        .await
        .wrap_err_with(|| format!("failed to download {source_url}"))?;
    let status = response.status();
    if !status.is_success() {
        bail!("model source returned HTTP {status}: {source_url}");
    }

    let (bytes, digest) = download_to_partial(
        response,
        &partial_path,
        model.archive_size_bytes,
        cancellation_token,
    )
    .await?;
    if digest != model.archive_sha256 {
        remove_partial(&partial_path).await;
        bail!(
            "model archive SHA-256 mismatch: catalog expects {}, received {digest}",
            model.archive_sha256
        );
    }

    tokio::fs::rename(&partial_path, &archive_path)
        .await
        .wrap_err_with(|| {
            format!(
                "failed to install verified archive {}",
                archive_path.display()
            )
        })?;

    let report = AcquisitionReport {
        model: model.id.to_string(),
        source: source.id.to_string(),
        source_display_name: source.display_name.to_string(),
        source_url,
        archive_path: archive_path.display().to_string(),
        acquisition_receipt_path: receipt_path.display().to_string(),
        bytes,
        sha256: digest,
        verified: true,
    };
    write_receipt(&receipt_path, &report).await?;
    Ok(report)
}

/// Download, verify, and atomically install a native model bundle archive.
///
/// # Errors
///
/// Returns an error when the source is unknown or unconfigured, the HTTP
/// request fails, cancellation is requested, or archive verification fails.
pub async fn acquire_native(
    model: ModelDefinition,
    source_selector: &str,
    cancellation_token: CancellationToken,
) -> Result<NativeAcquisitionReport> {
    let source = resolve_source(source_selector)?;
    let source_url = resolve_source_url(
        source.id,
        source.native_url_env_var,
        source.native_url,
        "native bundle",
    )?;

    let archive_path = model_registry::native_bundle_archive_path(model);
    let receipt_path = model_registry::native_bundle_acquisition_receipt_path(model);
    let parent = archive_path.parent().ok_or_else(|| {
        eyre!(
            "native bundle archive path has no parent: {}",
            archive_path.display()
        )
    })?;
    tokio::fs::create_dir_all(parent).await.wrap_err_with(|| {
        format!(
            "failed to create native bundle directory {}",
            parent.display()
        )
    })?;

    let partial_path = parent.join("native-bundle.zip.partial");
    let client = Client::new();
    let response = client
        .get(&source_url)
        .send()
        .await
        .wrap_err_with(|| format!("failed to download native bundle {source_url}"))?;
    let status = response.status();
    if !status.is_success() {
        bail!("native bundle source returned HTTP {status}: {source_url}");
    }

    let (bytes, digest) = download_to_partial(
        response,
        &partial_path,
        model.native_bundle_size_bytes,
        cancellation_token,
    )
    .await?;
    if digest != model.native_bundle_sha256 {
        remove_partial(&partial_path).await;
        bail!(
            "native bundle archive SHA-256 mismatch: catalog expects {}, received {digest}",
            model.native_bundle_sha256
        );
    }

    tokio::fs::rename(&partial_path, &archive_path)
        .await
        .wrap_err_with(|| {
            format!(
                "failed to install verified native bundle archive {}",
                archive_path.display()
            )
        })?;

    let report = NativeAcquisitionReport {
        model: model.id.to_string(),
        source: source.id.to_string(),
        source_display_name: source.display_name.to_string(),
        source_url,
        archive_path: archive_path.display().to_string(),
        acquisition_receipt_path: receipt_path.display().to_string(),
        bytes,
        sha256: digest,
        verified: true,
    };
    write_native_receipt(&receipt_path, &report).await?;
    Ok(report)
}

async fn download_to_partial(
    response: reqwest::Response,
    partial_path: &Path,
    expected_archive_size: u64,
    cancellation_token: CancellationToken,
) -> Result<(u64, String)> {
    let expected_content_length = response.content_length();
    let mut stream = response.bytes_stream();
    let mut output = tokio::fs::File::create(partial_path)
        .await
        .wrap_err_with(|| {
            format!(
                "failed to create partial archive {}",
                partial_path.display()
            )
        })?;
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;

    while let Some(chunk) = stream.next().await {
        cancellation_token.bail_if_cancelled()?;
        let chunk = chunk.wrap_err("failed while reading model archive response")?;
        hasher.update(&chunk);
        output
            .write_all(&chunk)
            .await
            .wrap_err("failed while writing partial model archive")?;
        bytes = bytes
            .checked_add(u64::try_from(chunk.len()).wrap_err("chunk length overflow")?)
            .ok_or_else(|| eyre!("model archive byte count overflow"))?;
    }
    output
        .flush()
        .await
        .wrap_err("failed to flush model archive")?;
    output
        .sync_all()
        .await
        .wrap_err("failed to sync model archive")?;
    drop(output);

    if expected_content_length.is_some_and(|expected| expected != bytes) {
        remove_partial(partial_path).await;
        bail!("model archive HTTP size mismatch: received {bytes} bytes");
    }
    if bytes != expected_archive_size {
        remove_partial(partial_path).await;
        bail!(
            "model archive size mismatch: catalog expects {expected_archive_size} bytes, received {bytes}"
        );
    }

    Ok((bytes, hex_digest(&hasher.finalize())))
}

fn resolve_source(selector: &str) -> Result<SourceDefinition> {
    let normalized = selector
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    SOURCES
        .iter()
        .copied()
        .find(|source| {
            source
                .id
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .flat_map(char::to_lowercase)
                .eq(normalized.chars())
        })
        .ok_or_else(|| {
            let known = SOURCES
                .iter()
                .map(|source| source.id)
                .collect::<Vec<_>>()
                .join(", ");
            eyre!("unknown model source '{selector}'; known sources: {known}")
        })
}

fn resolve_source_url(
    source_id: &str,
    environment_variable: &str,
    default_url: Option<&str>,
    artifact_description: &str,
) -> Result<String> {
    match std::env::var(environment_variable) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => bail!("{environment_variable} cannot be empty"),
        Err(std::env::VarError::NotPresent) => default_url.map(str::to_owned).ok_or_else(|| {
            eyre!(
                "{artifact_description} source {source_id} is not configured; set {environment_variable} to its HTTPS archive URL"
            )
        }),
        Err(error) => Err(error).wrap_err_with(|| {
            format!("failed to read {environment_variable} for source {source_id}")
        }),
    }
}

async fn write_receipt(path: &Path, report: &AcquisitionReport) -> Result<()> {
    let contents =
        facet_json::to_string_pretty(report).wrap_err("failed to serialize acquisition receipt")?;
    let temporary_path = path.with_file_name("acquisition.json.partial");
    tokio::fs::write(&temporary_path, contents)
        .await
        .wrap_err_with(|| {
            format!(
                "failed to write acquisition receipt {}",
                temporary_path.display()
            )
        })?;
    tokio::fs::rename(&temporary_path, path)
        .await
        .wrap_err_with(|| format!("failed to install acquisition receipt {}", path.display()))?;
    Ok(())
}

async fn write_native_receipt(path: &Path, report: &NativeAcquisitionReport) -> Result<()> {
    let contents = facet_json::to_string_pretty(report)
        .wrap_err("failed to serialize native acquisition receipt")?;
    let temporary_path = path.with_file_name("acquisition.json.partial");
    tokio::fs::write(&temporary_path, contents)
        .await
        .wrap_err_with(|| {
            format!(
                "failed to write native acquisition receipt {}",
                temporary_path.display()
            )
        })?;
    tokio::fs::rename(&temporary_path, path)
        .await
        .wrap_err_with(|| {
            format!(
                "failed to install native acquisition receipt {}",
                path.display()
            )
        })?;
    Ok(())
}

async fn remove_partial(path: &Path) {
    let _ = tokio::fs::remove_file(path).await;
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::resolve_source;

    #[test]
    fn source_selectors_are_case_and_separator_insensitive() {
        assert_eq!(
            resolve_source("Teamy").expect("Teamy should resolve").id,
            "Teamy"
        );
        assert_eq!(
            resolve_source("r2d2fish-onedrive")
                .expect("R2D2FISH-OneDrive should resolve")
                .id,
            "R2D2FISH-OneDrive"
        );
    }

    #[test]
    fn teamy_source_has_baked_content_addressed_urls() {
        let source = resolve_source("Teamy").expect("Teamy should resolve");

        assert!(
            source
                .raw_url
                .is_some_and(|url| url.contains("/raw/glados/"))
        );
        assert!(
            source
                .native_url
                .is_some_and(|url| url.contains("/native/glados/"))
        );
    }
}
