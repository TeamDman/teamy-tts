use crate::cli::output::CliOutput;
use crate::model_registry;
use crate::model_sources;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use teamy_cancellation::CancellationToken;

/// Acquire the raw archive for the first known model.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelAcquireUnpreparedArgs {
    /// Source selector: `Teamy` or `R2D2FISH-OneDrive`.
    #[facet(args::positional)]
    pub source: String,
}

impl ModelAcquireUnpreparedArgs {
    /// # Errors
    ///
    /// Returns an error when the source is unknown, its URL is not configured,
    /// the download fails, or archive verification fails.
    pub async fn invoke(self, cancellation_token: CancellationToken) -> Result<CliOutput> {
        let Some(model) = model_registry::find_model("glados") else {
            bail!("the built-in glados model is missing from the model catalog");
        };
        let report = model_sources::acquire(model, &self.source, cancellation_token).await?;
        Ok(CliOutput::facet(report))
    }
}
