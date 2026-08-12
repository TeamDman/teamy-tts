use crate::cli::backend::benchmark::BackendBenchmarkArgs;
use crate::cli::backend::list::BackendListArgs;
use crate::cli::backend::probe::BackendProbeArgs;
use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;

/// Backend discovery and profiling commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct BackendArgs {
    /// The backend subcommand to run.
    #[facet(args::subcommand)]
    pub command: BackendCommand,
}

/// Backend subcommands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum BackendCommand {
    /// List backend availability and automatic-selection evidence.
    List(BackendListArgs),
    /// Benchmark a backend against the stable correctness corpus.
    Benchmark(BackendBenchmarkArgs),
    /// Probe the optional Vulkan compute substrate.
    Probe(BackendProbeArgs),
}

impl BackendArgs {
    /// # Errors
    ///
    /// Returns an error when the selected backend command fails.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            BackendCommand::List(args) => args.invoke().await,
            BackendCommand::Benchmark(args) => args.invoke().await,
            BackendCommand::Probe(args) => args.invoke().await,
        }
    }
}
