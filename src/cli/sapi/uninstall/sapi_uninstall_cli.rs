use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct UninstallArgs {
    /// Remove this user's explicit-client registration instead of the machine voice.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub per_user: bool,
}
impl UninstallArgs {
    pub fn invoke(self) -> eyre::Result<CliOutput> {
        #[cfg(all(windows, feature = "cuda-native"))]
        {
            teamy_tts_sapi::registration::uninstall_scope(false, true, !self.per_user)?;
            teamy_tts_ipc::client::stop("main")?;
            Ok(CliOutput::facet("Removed Teamy SAPI registration. Close applications that still have the old voice loaded.".to_string()))
        }
        #[cfg(not(all(windows, feature = "cuda-native")))]
        {
            let _ = self;
            eyre::bail!("SAPI requires a Windows cuda-native build")
        }
    }
}
