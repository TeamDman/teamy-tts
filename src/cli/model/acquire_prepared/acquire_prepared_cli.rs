use crate::cli::output::CliOutput;
use crate::model_registry;
use crate::model_sources;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use teamy_cancellation::CancellationToken;

/// Acquire and install the verified native bundle for the first known model.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelAcquirePreparedArgs {
    /// Source selector: `Teamy` or `R2D2FISH-OneDrive`.
    #[facet(args::positional)]
    pub source: String,
}

impl ModelAcquirePreparedArgs {
    /// # Errors
    ///
    /// Returns an error when the source is unknown, its URL is not configured,
    /// the download fails, archive verification fails, or native installation
    /// fails.
    pub async fn invoke(self, cancellation_token: CancellationToken) -> Result<CliOutput> {
        let Some(model) = model_registry::find_model("glados") else {
            bail!("the built-in glados model is missing from the model catalog");
        };
        let report = model_sources::acquire_native(model, &self.source, cancellation_token).await?;
        tracing::info!(source = %self.source, "installing acquired native model");
        let prepared = model_registry::prepare_native_bundle_archive(
            model,
            &PathBuf::from(&report.archive_path),
            true,
        )?;

        Ok(CliOutput::facet(ModelAcquirePreparedReport {
            model: report.model,
            source: report.source,
            source_display_name: report.source_display_name,
            source_url: report.source_url,
            archive_path: report.archive_path,
            acquisition_receipt_path: report.acquisition_receipt_path,
            bytes: report.bytes,
            sha256: report.sha256,
            verified: report.verified,
            prepared_dir: prepared.root.display().to_string(),
            artifact_count: prepared.manifest.artifacts.len(),
        }))
    }
}

#[derive(Facet, Debug)]
struct ModelAcquirePreparedReport {
    model: String,
    source: String,
    source_display_name: String,
    source_url: String,
    archive_path: String,
    acquisition_receipt_path: String,
    bytes: u64,
    sha256: String,
    verified: bool,
    prepared_dir: String,
    artifact_count: usize,
}
