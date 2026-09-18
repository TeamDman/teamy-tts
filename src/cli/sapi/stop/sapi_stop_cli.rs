use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct StopArgs {
    /// Worker instance to stop. Defaults to main.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub instance: Option<String>,
}
impl StopArgs {
    pub fn invoke(self) -> eyre::Result<CliOutput> {
        #[cfg(all(windows, feature = "cuda-native"))]
        {
            teamy_tts_ipc::client::stop(self.instance.as_deref().unwrap_or("main"))?;
            Ok(CliOutput::facet("Worker stopped.".to_string()))
        }
        #[cfg(not(all(windows, feature = "cuda-native")))]
        {
            let _ = self;
            eyre::bail!("SAPI requires a Windows cuda-native build")
        }
    }
}
