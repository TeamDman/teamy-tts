use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct InstallArgs {
    /// Release SAPI DLL path. Defaults to the adapter selected by the installer.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub dll: Option<String>,
    /// Register only for this user (explicit clients only; absent from Windows voice list).
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub per_user: bool,
}
impl InstallArgs {
    pub fn invoke(self) -> eyre::Result<CliOutput> {
        #[cfg(all(windows, feature = "cuda-native"))]
        {
            let worker = std::env::current_exe()?.canonicalize()?;
            let installed_selection = worker
                .parent()
                .and_then(std::path::Path::parent)
                .map(|root| root.join("share/teamy-tts/sapi/current.txt"))
                .and_then(|path| std::fs::read_to_string(path).ok())
                .map(|value| std::path::PathBuf::from(value.trim()));
            let dll = self
                .dll
                .map(std::path::PathBuf::from)
                .or(installed_selection)
                .unwrap_or_else(|| worker.with_file_name("teamy_tts_sapi.dll"))
                .canonicalize()?;
            // Machine-wide COM registration must not load executable code from
            // the user's writable Cargo directory into an elevated host.
            let dll = if self.per_user {
                dll
            } else {
                use sha2::{Digest, Sha256};
                let bytes = std::fs::read(&dll)?;
                let hash = format!("{:x}", Sha256::digest(&bytes));
                let program_files = std::env::var_os("ProgramW6432")
                    .or_else(|| std::env::var_os("ProgramFiles"))
                    .ok_or_else(|| eyre::eyre!("Program Files directory is unavailable"))?;
                let directory = std::path::PathBuf::from(program_files)
                    .join("Teamy TTS")
                    .join("sapi")
                    .join(hash);
                std::fs::create_dir_all(&directory).map_err(|e| {
                    eyre::eyre!("Machine SAPI installation requires an administrator terminal: {e}")
                })?;
                let target = directory.join("teamy_tts_sapi.dll");
                if target.exists() {
                    eyre::ensure!(
                        std::fs::read(&target)? == bytes,
                        "Installed SAPI DLL checksum mismatch"
                    );
                } else {
                    use std::io::Write;
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&target)?;
                    file.write_all(&bytes)?;
                    file.sync_all()?;
                }
                target
            };
            teamy_tts_sapi::registration::install_scope(&dll, &worker, false, !self.per_user)
                .map_err(|e| eyre::eyre!("SAPI registration failed: {e}. Machine registration requires an administrator terminal; --per-user is available for explicit clients."))?;
            Ok(CliOutput::facet(format!(
                "Registered Teamy TTS ({}) with {}. Default voice unchanged.",
                if self.per_user { "per-user" } else { "machine" },
                dll.display()
            )))
        }
        #[cfg(not(all(windows, feature = "cuda-native")))]
        {
            let _ = self;
            eyre::bail!("SAPI requires a Windows cuda-native build")
        }
    }
}
