use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
use teamy_cancellation::CancellationToken;

#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ServeArgs {
    /// Worker instance within this Windows user/session. Defaults to main.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub instance: Option<String>,
    /// Exit after this many seconds without clients. Defaults to 120 (range 1..86400).
    #[facet(args::named)]
    #[arbitrary(default)]
    pub idle_seconds: Option<u32>,
}
impl ServeArgs {
    pub async fn invoke(self, cancellation: CancellationToken) -> eyre::Result<CliOutput> {
        #[cfg(all(windows, feature = "cuda-native"))]
        {
            let idle = self.idle_seconds.unwrap_or(120);
            eyre::ensure!(
                (1..=86400).contains(&idle),
                "idle-seconds must be in 1..=86400"
            );
            crate::worker::serve(
                self.instance.as_deref().unwrap_or("main"),
                idle,
                cancellation,
            )?;
            Ok(CliOutput::none())
        }
        #[cfg(not(all(windows, feature = "cuda-native")))]
        {
            let _ = (self, cancellation);
            eyre::bail!("serve requires a Windows cuda-native build")
        }
    }
}
