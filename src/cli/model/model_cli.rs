use super::ModelAcquirePreparedArgs;
use super::ModelAcquireUnpreparedArgs;
use super::ModelPrepareArgs;
use crate::cli::output::CliOutput;
use crate::model_registry;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use teamy_cancellation::CancellationToken;

/// Model acquisition and preparation commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelArgs {
    /// The model subcommand to run.
    #[facet(args::subcommand)]
    pub command: ModelCommand,
}

/// Model subcommands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum ModelCommand {
    /// Download, verify, and install a native model bundle from a named source.
    AcquirePrepared(ModelAcquirePreparedArgs),
    /// Download and verify a raw model archive from a named source.
    AcquireUnprepared(ModelAcquireUnpreparedArgs),
    /// Install a native model bundle from a local converter or archive.
    Prepare(ModelPrepareArgs),
    /// List the known models and their local installation state.
    List(ModelListArgs),
    /// Show one model's manifest requirements and local paths.
    Show(ModelShowArgs),
}

/// List known models.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelListArgs;

/// Show one known model.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelShowArgs {
    /// Stable model identifier, such as `glados`.
    #[facet(args::positional)]
    pub model: String,
}

#[derive(Facet, Debug)]
struct ModelListReport {
    models: Vec<model_registry::ModelReport>,
}

impl ModelArgs {
    /// # Errors
    ///
    /// Returns an error if a model command cannot resolve its requested model.
    pub async fn invoke(self, cancellation_token: CancellationToken) -> Result<CliOutput> {
        match self.command {
            ModelCommand::AcquirePrepared(args) => args.invoke(cancellation_token).await,
            ModelCommand::AcquireUnprepared(args) => args.invoke(cancellation_token).await,
            ModelCommand::Prepare(args) => args.invoke().await,
            ModelCommand::List(args) => args.invoke().await,
            ModelCommand::Show(args) => args.invoke().await,
        }
    }
}

impl ModelListArgs {
    /// # Errors
    ///
    /// This command does not currently perform fallible I/O.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let models = model_registry::all_models()
            .iter()
            .copied()
            .map(model_registry::report_for)
            .collect::<Result<Vec<_>>>()?;
        Ok(CliOutput::facet(ModelListReport { models }))
    }
}

impl ModelShowArgs {
    /// # Errors
    ///
    /// Returns an error when the model identifier is not in the catalog.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let Some(model) = model_registry::find_model(&self.model) else {
            let known = model_registry::all_models()
                .iter()
                .map(|model| model.id)
                .collect::<Vec<_>>()
                .join(", ");
            bail!("unknown model '{}'; known models: {known}", self.model);
        };
        Ok(CliOutput::facet(model_registry::report_for(model)?))
    }
}
