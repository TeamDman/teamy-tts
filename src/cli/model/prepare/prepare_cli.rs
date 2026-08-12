use crate::cli::output::CliOutput;
use crate::model_registry;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

/// Install a converter-produced native model bundle.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelPrepareArgs {
    /// Stable model identifier, such as `glados`.
    #[facet(args::positional)]
    pub model: String,

    /// Directory containing the fixed native bundle artifact names.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub source_dir: Option<String>,

    /// ZIP archive containing the fixed native bundle artifact names.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub source_archive: Option<String>,

    /// Update an existing prepared revision in place.
    #[facet(args::named, default)]
    pub force: bool,
}

#[derive(Facet, Debug)]
struct ModelPrepareReport {
    model: String,
    source: String,
    prepared_dir: String,
    artifact_count: usize,
}

impl ModelPrepareArgs {
    /// # Errors
    ///
    /// Returns an error if the model is unknown or the native bundle cannot be
    /// installed and verified.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_str();
        let Some(model) = model_registry::find_model(model_id) else {
            bail!("unknown model {model_id:?}; known models: glados");
        };
        if self.source_dir.is_some() && self.source_archive.is_some() {
            bail!("choose only one of --source-dir or --source-archive");
        }
        let (source, artifacts) = match (self.source_dir, self.source_archive) {
            (Some(_), Some(_)) => {
                bail!("choose only one of --source-dir or --source-archive");
            }
            (Some(source_dir), None) => {
                let source_dir = PathBuf::from(source_dir);
                let artifacts =
                    model_registry::prepare_native_bundle(model, &source_dir, self.force)?;
                (source_dir, artifacts)
            }
            (None, Some(source_archive)) => {
                let source_archive = PathBuf::from(source_archive);
                let artifacts = model_registry::prepare_native_bundle_archive(
                    model,
                    &source_archive,
                    self.force,
                )?;
                (source_archive, artifacts)
            }
            (None, None) => {
                let source_archive = model_registry::native_bundle_archive_path(model);
                let receipt = model_registry::native_bundle_acquisition_receipt_path(model);
                if !source_archive.is_file() || !receipt.is_file() {
                    bail!(
                        "model prepare requires --source-dir or --source-archive; no verified native bundle is cached (run model acquire-prepared first)"
                    );
                }
                let artifacts = model_registry::prepare_native_bundle_archive(
                    model,
                    &source_archive,
                    self.force,
                )?;
                (source_archive, artifacts)
            }
        };
        Ok(CliOutput::facet(ModelPrepareReport {
            model: model.id.to_string(),
            source: source.display().to_string(),
            prepared_dir: artifacts.root.display().to_string(),
            artifact_count: artifacts.manifest.artifacts.len(),
        }))
    }
}
