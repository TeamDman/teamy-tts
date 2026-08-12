use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;

/// Probe Vulkan devices, queues, extensions, and cooperative-matrix support.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct BackendProbeArgs;

impl BackendProbeArgs {
    /// # Errors
    ///
    /// Returns an actionable error when Vulkan support was not compiled or
    /// when the loader/device probe fails.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        #[cfg(feature = "vulkan")]
        {
            Ok(CliOutput::facet(crate::vulkan::probe()?))
        }

        #[cfg(not(feature = "vulkan"))]
        {
            let _ = self;
            eyre::bail!(
                "Vulkan support is unavailable in this build; rebuild with --features vulkan"
            );
        }
    }
}
