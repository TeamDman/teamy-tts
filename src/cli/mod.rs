pub mod benchmark;
pub mod cache;
pub mod config;
pub mod facet_shape;
pub mod global_args;
pub mod home;
pub mod interactive;
pub mod model;
pub mod output;
pub mod say;

use crate::cli::benchmark::BenchmarkArgs;
use crate::cli::cache::CacheArgs;
use crate::cli::config::ConfigArgs;
use crate::cli::global_args::GlobalArgs;
use crate::cli::home::HomeArgs;
use crate::cli::interactive::InteractiveArgs;
use crate::cli::model::{ModelAcquirePreparedArgs, ModelArgs, ModelCommand, ModelPrepareArgs};
use crate::cli::output::CliOutput;
use crate::cli::say::{SayArgs, WriteArgs};
use crate::model_registry;
use arbitrary::Arbitrary;
use eyre::Context;
use facet::Facet;
use figue::FigueBuiltins;
use figue::ToArgs;
use figue::{self as args};
use teamy_cancellation::CancellationToken;

/// A local `GLaDOS` text-to-speech utility.
///
/// Environment variables:
/// - `TEAMY_TTS_HOME_DIR` overrides the resolved application home directory.
/// - `TEAMY_TTS_CACHE_DIR` overrides the resolved cache directory.
/// - `TEAMY_TTS_BACKEND`, `TEAMY_TTS_MODEL_DIR`, `TEAMY_TTS_TORCH_MODEL_DIR`,
///   and `TEAMY_TTS_TORCH_DEVICE` override remembered configuration values.
/// - `RUST_LOG` provides a tracing filter when `--log-filter` is omitted.
#[derive(Facet, Arbitrary, Debug)]
pub struct Cli {
    /// Global arguments (`debug`, `log_filter`, `log_file`).
    #[facet(flatten)]
    pub global_args: GlobalArgs,

    /// Standard CLI options (help, version, completions).
    #[facet(flatten)]
    #[arbitrary(default)]
    pub builtins: FigueBuiltins,

    /// The command to run.
    #[facet(args::subcommand)]
    pub command: Command,
}

impl PartialEq for Cli {
    fn eq(&self, other: &Self) -> bool {
        // Ignore builtins in comparison since FigueBuiltins doesn't implement PartialEq
        self.global_args == other.global_args && self.command == other.command
    }
}

impl Cli {
    /// # Errors
    ///
    /// This function will return an error if the tokio runtime cannot be built or if the command fails.
    pub fn invoke(self, cancellation_token: CancellationToken) -> eyre::Result<CliOutput> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .wrap_err("Failed to build tokio runtime")?;
        runtime.block_on(async move { self.command.invoke(cancellation_token).await })
    }
}

/// Render a command that can be pasted into the current shell.
///
/// Keeping this construction typed means a model-preparation hint follows the
/// same Figue schema as the command parser. The executable-prefixed form is
/// especially useful while running the development binary through `cargo run`.
fn render_command(command: Command) -> String {
    let cli = Cli {
        global_args: GlobalArgs::default(),
        builtins: FigueBuiltins::default(),
        command,
    };

    cli.to_args_string_with_current_exe().map_or_else(
        |error| {
            tracing::debug!(%error, "failed to render a command hint");
            "teamy-tts model prepare glados".to_string()
        },
        |value| value.to_string_lossy().into_owned(),
    )
}

/// Explain how to install a model that `say` needs.
pub(crate) fn model_preparation_hint(model: model_registry::ModelDefinition) -> String {
    let prepare = render_command(Command::Model(ModelArgs {
        command: ModelCommand::Prepare(ModelPrepareArgs {
            model: model.id.to_string(),
            source_dir: None,
            source_archive: None,
            force: false,
        }),
    }));

    let native_bundle_is_cached = model_registry::native_bundle_archive_path(model).is_file()
        && model_registry::native_bundle_acquisition_receipt_path(model).is_file();
    if native_bundle_is_cached {
        return format!("Prepare the verified native bundle with:\n  {prepare}");
    }

    let acquire = render_command(Command::Model(ModelArgs {
        command: ModelCommand::AcquirePrepared(ModelAcquirePreparedArgs {
            source: "Teamy".to_string(),
        }),
    }));
    format!("Acquire and install the verified native bundle with:\n  {acquire}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_preparation_hint_uses_the_cli_schema() {
        let model = model_registry::find_model("glados").expect("catalog should contain glados");
        let hint = model_preparation_hint(model);

        let native_bundle_is_cached = model_registry::native_bundle_archive_path(model).is_file()
            && model_registry::native_bundle_acquisition_receipt_path(model).is_file();
        if native_bundle_is_cached {
            assert!(hint.contains("model prepare glados"));
            assert!(!hint.contains("model acquire-prepared Teamy"));
        } else {
            assert!(!hint.contains("model prepare glados"));
            assert!(hint.contains("model acquire-prepared Teamy"));
        }
    }
}

/// Top-level teamy-tts commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum Command {
    /// Synthesize text, write a WAV file, and play it.
    Say(SayArgs),
    /// Synthesize text into a WAV file without playing it.
    Write(WriteArgs),
    /// Measure cold-load and warm synthesis latency.
    Benchmark(BenchmarkArgs),
    /// Read stdin lines, write each WAV file, and play each one.
    Interactive(InteractiveArgs),
    /// Cache-related commands.
    Cache(CacheArgs),
    /// Durable configuration and environment overrides.
    Config(ConfigArgs),
    /// Home-related commands.
    Home(HomeArgs),
    /// Model acquisition and preparation commands.
    Model(ModelArgs),
}

impl Command {
    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(self, cancellation_token: CancellationToken) -> eyre::Result<CliOutput> {
        cancellation_token.bail_if_cancelled()?;
        match self {
            Command::Say(args) => args.invoke().await,
            Command::Write(args) => args.invoke().await,
            Command::Benchmark(args) => args.invoke().await,
            Command::Interactive(args) => args.invoke(cancellation_token).await,
            Command::Cache(args) => args.invoke().await,
            Command::Config(args) => args.invoke().await,
            Command::Home(args) => args.invoke().await,
            Command::Model(args) => args.invoke(cancellation_token).await,
        }
    }
}
