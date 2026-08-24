//! Inspect the phoneme sequence produced by the local `GLaDOS` frontend.

use crate::cli::model_preparation_hint;
use crate::cli::output::CliOutput;
use crate::config;
use crate::model_registry;
use crate::runtime::GladosTextFrontend;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::time::Instant;

/// The typed output from the `phonemize` command.
#[derive(Clone, Debug, Facet, PartialEq)]
#[facet(rename_all = "kebab-case")]
pub struct PhonemizeReport {
    pub model: String,
    pub phonemes: String,
    pub token_ids: Vec<i32>,
}

/// Convert ordinary text to the exact `GLaDOS` phoneme sequence and token IDs.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct PhonemizeArgs {
    /// Text to normalize and phonemize.
    #[facet(args::positional)]
    pub text: String,

    /// Stable model identifier. Defaults to `glados`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,
}

impl PhonemizeArgs {
    /// # Errors
    ///
    /// Returns an error if the selected model is unknown or unprepared, the
    /// frontend artifacts cannot be loaded, or phonemization fails.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let (model, frontend) = load_text_frontend(model_id)?;
        let started = Instant::now();
        let phonemes = frontend.phonemize(&self.text)?;
        let token_ids = frontend.tokenize_phonemes(&phonemes)?;
        tracing::info!(
            model = %model.id,
            token_count = token_ids.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "phonemization complete"
        );

        Ok(CliOutput::facet(PhonemizeReport {
            model: model.id.to_string(),
            phonemes,
            token_ids,
        }))
    }
}

/// Load only the dictionary and neural phonemizer, without loading the
/// acoustic model or vocoder.
pub(crate) fn load_text_frontend(
    model_id: &str,
) -> Result<(model_registry::ModelDefinition, GladosTextFrontend)> {
    let Some(model) = model_registry::find_model(model_id) else {
        bail!("unknown model {model_id:?}; known models: glados");
    };
    tracing::info!(model = %model.id, "loading prepared text frontend");
    let prepared = model_registry::inspect_prepared_model_dir(model).wrap_err_with(|| {
        format!(
            "model {:?} is not prepared at the required location; {}",
            model.id,
            model_preparation_hint(model)
        )
    })?;
    let Some(model_dir) = config::effective_torch_model_dir()? else {
        bail!(
            "tch/LibTorch model directory is not configured; set it once with `teamy-tts config set --torch-model-dir <path>`"
        );
    };
    let frontend = GladosTextFrontend::from_prepared(&prepared, &model_dir)?;
    Ok((model, frontend))
}
