use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct StatusArgs {
    /// Inspect this user's explicit-client registration instead of the machine voice.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub per_user: bool,
}
impl StatusArgs {
    pub fn invoke(self) -> eyre::Result<CliOutput> {
        #[cfg(all(windows, feature = "cuda-native"))]
        {
            let machine = !self.per_user;
            let installed = teamy_tts_sapi::registration::installed(machine)?;
            let voice = teamy_tts_sapi::host::status(machine)?;
            let (worker_state, worker_pid) = match teamy_tts_ipc::client::status("main") {
                Ok(status) => (
                    format!("{} {}", status.state, status.detail)
                        .trim()
                        .to_string(),
                    Some(status.pid),
                ),
                Err(e) if e.raw_os_error() == Some(2) => ("stopped".into(), None),
                Err(e) => (format!("unavailable: {e}"), None),
            };
            Ok(CliOutput::facet(SapiStatusReport {
                registered: installed.is_some(),
                enumerated: voice.enumerated,
                is_default: voice.is_default,
                default_voice: voice.default_voice,
                dll_path: installed.as_ref().map(|i| i.dll.clone()),
                worker_path: installed.as_ref().map(|i| i.worker.clone()),
                dll_exists: installed
                    .as_ref()
                    .is_some_and(|i| std::path::Path::new(&i.dll).is_file()),
                worker_exists: installed
                    .as_ref()
                    .is_some_and(|i| std::path::Path::new(&i.worker).is_file()),
                worker_state,
                worker_pid,
                protocol: 1,
                model_dir: crate::config::effective_native_model_dir()?
                    .map(|p| p.display().to_string()),
            }))
        }
        #[cfg(not(all(windows, feature = "cuda-native")))]
        {
            let _ = self;
            eyre::bail!("SAPI requires a Windows cuda-native build")
        }
    }
}
#[derive(Facet, Debug)]
struct SapiStatusReport {
    registered: bool,
    enumerated: bool,
    is_default: bool,
    default_voice: String,
    dll_path: Option<String>,
    worker_path: Option<String>,
    dll_exists: bool,
    worker_exists: bool,
    worker_state: String,
    worker_pid: Option<u32>,
    protocol: u16,
    model_dir: Option<String>,
}
