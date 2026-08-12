//! Commands for inspecting and changing durable teamy-tts settings.

use crate::cli::output::CliOutput;
use crate::config::EffectiveConfig;
use crate::config::StoredConfig;
use crate::config::config_path;
use crate::config::effective;
use crate::config::load;
use crate::config::save;
use arbitrary::Arbitrary;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;

/// Configuration commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ConfigArgs {
    /// The configuration subcommand to run.
    #[facet(args::subcommand)]
    pub command: ConfigCommand,
}

/// Configuration subcommands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum ConfigCommand {
    /// Show stored and effective settings.
    Show(ConfigShowArgs),
    /// Set one or more remembered settings.
    Set(ConfigSetArgs),
    /// Clear one or more remembered settings.
    Clear(ConfigClearArgs),
}

/// Show configuration settings.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ConfigShowArgs;

/// Set remembered configuration settings.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[facet(rename_all = "kebab-case")]
pub struct ConfigSetArgs {
    /// Compatibility backend spelling: tch, torchscript, or libtorch.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub backend: Option<String>,

    /// Root containing prepared native model revisions.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model_dir: Option<String>,

    /// Directory containing the upstream `TorchScript` model files.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub torch_model_dir: Option<String>,

    /// CUDA device index used by `LibTorch`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub torch_device: Option<i32>,
}

/// Clear remembered configuration settings.
#[expect(
    clippy::struct_excessive_bools,
    reason = "Each flag independently selects one durable setting to clear."
)]
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[facet(rename_all = "kebab-case")]
pub struct ConfigClearArgs {
    /// Clear the remembered backend policy.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub backend: bool,

    /// Clear the remembered prepared-model root.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub model_dir: bool,

    /// Clear the remembered `TorchScript` model directory.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub torch_model_dir: bool,

    /// Clear the remembered `LibTorch` CUDA device.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub torch_device: bool,
}

#[derive(Facet, Debug)]
struct ConfigReport {
    path: String,
    stored: StoredConfig,
    effective: EffectiveConfig,
}

impl ConfigArgs {
    /// # Errors
    ///
    /// Returns an error when configuration cannot be read, validated, or
    /// written.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            ConfigCommand::Show(args) => args.invoke().await,
            ConfigCommand::Set(args) => args.invoke().await,
            ConfigCommand::Clear(args) => args.invoke().await,
        }
    }
}

impl ConfigShowArgs {
    /// # Errors
    ///
    /// Returns an error when configuration cannot be read or resolved.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        Ok(CliOutput::facet(report()?))
    }
}

impl ConfigSetArgs {
    /// # Errors
    ///
    /// Returns an error when a setting is invalid or the file cannot be
    /// written.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        if self.backend.is_none()
            && self.model_dir.is_none()
            && self.torch_model_dir.is_none()
            && self.torch_device.is_none()
        {
            bail!(
                "config set requires at least one setting; use --backend, --model-dir, --torch-model-dir, or --torch-device"
            );
        }

        let mut stored = load()?;
        if let Some(backend) = self.backend {
            let backend = crate::backend::BackendSelection::parse(Some(&backend))?;
            stored.backend = Some(backend.to_string());
        }
        if let Some(model_dir) = self.model_dir {
            stored.model_dir = Some(non_empty_path("--model-dir", model_dir)?);
        }
        if let Some(torch_model_dir) = self.torch_model_dir {
            stored.torch_model_dir = Some(non_empty_path("--torch-model-dir", torch_model_dir)?);
        }
        if let Some(torch_device) = self.torch_device {
            stored.torch_device = Some(torch_device);
        }
        save(&stored)?;
        Ok(CliOutput::facet(report()?))
    }
}

impl ConfigClearArgs {
    /// # Errors
    ///
    /// Returns an error when no setting was selected or the file cannot be
    /// written.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        if !self.backend && !self.model_dir && !self.torch_model_dir && !self.torch_device {
            bail!(
                "config clear requires at least one setting; use --backend, --model-dir, --torch-model-dir, or --torch-device"
            );
        }

        let mut stored = load()?;
        if self.backend {
            stored.backend = None;
        }
        if self.model_dir {
            stored.model_dir = None;
        }
        if self.torch_model_dir {
            stored.torch_model_dir = None;
        }
        if self.torch_device {
            stored.torch_device = None;
        }
        save(&stored)?;
        Ok(CliOutput::facet(report()?))
    }
}

fn report() -> Result<ConfigReport> {
    Ok(ConfigReport {
        path: config_path().display().to_string(),
        stored: load()?,
        effective: effective()?,
    })
}

fn non_empty_path(flag: &str, value: String) -> Result<String> {
    if value.trim().is_empty() {
        bail!("{flag} cannot be empty");
    }
    let path = std::path::PathBuf::from(value);
    let path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    };
    Ok(path.display().to_string())
}
