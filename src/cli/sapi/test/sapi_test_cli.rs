use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct TestArgs {
    /// Text to render through real SAPI. Defaults to a short greeting.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub text: Option<String>,
    /// Capture SAPI audio to this WAV path; omit to play it.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub output: Option<String>,
    /// Use this user's explicit-client voice registration.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub per_user: bool,
}
impl TestArgs {
    pub fn invoke(self) -> eyre::Result<CliOutput> {
        #[cfg(all(windows, feature = "cuda-native"))]
        {
            let elapsed = teamy_tts_sapi::host::speak(
                !self.per_user,
                self.text
                    .as_deref()
                    .unwrap_or("Hello, friend. Teamy speech is ready."),
                self.output.as_deref().map(std::path::Path::new),
            )?;
            Ok(CliOutput::facet(format!(
                "SAPI synthesis completed in {} ms{}",
                elapsed.as_millis(),
                self.output
                    .map_or_else(String::new, |p| format!("; wrote {p}"))
            )))
        }
        #[cfg(not(all(windows, feature = "cuda-native")))]
        {
            let _ = self;
            eyre::bail!("SAPI requires a Windows cuda-native build")
        }
    }
}
