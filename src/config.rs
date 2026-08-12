//! Durable user configuration and environment-variable overrides.

use crate::backend::BackendSelection;
use crate::paths::APP_HOME;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Name of the durable configuration file under the application home.
pub const CONFIG_FILE_NAME: &str = "config.json";

/// Override the default backend policy.
pub const BACKEND_ENV_VAR: &str = "TEAMY_TTS_BACKEND";
/// Override the root containing prepared model revisions.
pub const MODEL_DIR_ENV_VAR: &str = "TEAMY_TTS_MODEL_DIR";
/// Override the upstream `TorchScript` model directory.
pub const TORCH_MODEL_DIR_ENV_VAR: &str = "TEAMY_TTS_TORCH_MODEL_DIR";
/// Override the CUDA device used by the `LibTorch` backend.
pub const TORCH_DEVICE_ENV_VAR: &str = "TEAMY_TTS_TORCH_DEVICE";

/// Values written to the durable configuration file.
#[derive(Clone, Debug, Default, Facet, PartialEq, Eq)]
pub struct StoredConfig {
    /// Default backend policy for synthesis commands.
    pub backend: Option<String>,
    /// Root containing prepared native model revisions.
    pub model_dir: Option<String>,
    /// Directory containing `glados-new.pt` and `vocoder-gpu.pt`.
    pub torch_model_dir: Option<String>,
    /// CUDA device index used by `LibTorch`.
    pub torch_device: Option<i32>,
}

/// Effective settings after applying environment overrides.
#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct EffectiveConfig {
    /// Backend policy used when a command does not pass `--backend`.
    pub backend: String,
    /// Prepared-model root, including its built-in default.
    pub model_dir: String,
    /// Effective `TorchScript` model directory, if configured.
    pub torch_model_dir: Option<String>,
    /// Effective CUDA device index, if configured; `LibTorch` defaults to zero.
    pub torch_device: Option<i32>,
}

/// Return the path to the durable configuration file.
#[must_use]
pub fn config_path() -> PathBuf {
    APP_HOME.file_path(CONFIG_FILE_NAME)
}

/// Load durable settings, treating a missing file as an empty configuration.
///
/// # Errors
///
/// Returns an error when the file exists but cannot be read or decoded.
pub fn load() -> Result<StoredConfig> {
    let path = config_path();
    if !path.is_file() {
        return Ok(StoredConfig::default());
    }

    let contents = fs::read_to_string(&path)
        .wrap_err_with(|| format!("failed to read configuration {}", path.display()))?;
    facet_json::from_str(&contents)
        .wrap_err_with(|| format!("failed to parse configuration {}", path.display()))
}

/// Persist durable settings and return the written path.
///
/// # Errors
///
/// Returns an error when the application home cannot be created or the file
/// cannot be serialized or written.
pub fn save(config: &StoredConfig) -> Result<PathBuf> {
    APP_HOME.ensure_dir()?;
    let path = config_path();
    let contents = facet_json::to_string_pretty(config)
        .wrap_err("failed to serialize teamy-tts configuration")?;
    fs::write(&path, format!("{contents}\n"))
        .wrap_err_with(|| format!("failed to write configuration {}", path.display()))?;
    Ok(path)
}

/// Resolve the backend policy with command-line, environment, and durable
/// configuration precedence.
///
/// The explicit command-line value wins over the environment, which wins over
/// the remembered value. With none of those set, the policy is `auto`.
///
/// # Errors
///
/// Returns an error when a configured backend value is invalid.
pub fn effective_backend(command_line: Option<&str>) -> Result<BackendSelection> {
    if command_line.is_some() {
        return BackendSelection::parse(command_line);
    }

    let stored = load()?;
    let environment = environment_value(BACKEND_ENV_VAR)?;
    BackendSelection::parse(environment.as_deref().or(stored.backend.as_deref()))
}

/// Resolve the prepared-model root override, without applying its built-in
/// cache default.
///
/// # Errors
///
/// Returns an error when an environment override is empty or configuration
/// cannot be loaded.
pub fn effective_model_dir() -> Result<Option<PathBuf>> {
    let stored = load()?;
    Ok(environment_value(MODEL_DIR_ENV_VAR)?
        .or(stored.model_dir)
        .map(PathBuf::from))
}

/// Resolve the upstream `TorchScript` model directory.
///
/// # Errors
///
/// Returns an error when an environment override is empty or configuration
/// cannot be loaded.
pub fn effective_torch_model_dir() -> Result<Option<PathBuf>> {
    let stored = load()?;
    Ok(environment_value(TORCH_MODEL_DIR_ENV_VAR)?
        .or(stored.torch_model_dir)
        .map(PathBuf::from))
}

/// Resolve the `LibTorch` CUDA device index.
///
/// # Errors
///
/// Returns an error when an environment override is not a signed 32-bit
/// integer or configuration cannot be loaded.
pub fn effective_torch_device() -> Result<Option<i32>> {
    if let Some(value) = environment_value(TORCH_DEVICE_ENV_VAR)? {
        return Ok(Some(value.parse::<i32>().wrap_err_with(|| {
            format!("{TORCH_DEVICE_ENV_VAR} is not an integer: {value:?}")
        })?));
    }

    Ok(load()?.torch_device)
}

/// Resolve all settings for `config show`.
///
/// # Errors
///
/// Returns an error when any configured value is invalid or a default path
/// cannot be resolved.
pub fn effective() -> Result<EffectiveConfig> {
    let backend = effective_backend(None)?.to_string();
    let model_dir = effective_model_dir()?.unwrap_or_else(|| {
        crate::paths::CacheHome::resolve()
            .map_or_else(|_| PathBuf::from("models"), |cache| cache.0.join("models"))
    });
    Ok(EffectiveConfig {
        backend,
        model_dir: model_dir.display().to_string(),
        torch_model_dir: effective_torch_model_dir()?.map(|path| path.display().to_string()),
        torch_device: effective_torch_device()?,
    })
}

fn environment_value(name: &str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) if value.trim().is_empty() => bail!("{name} cannot be empty"),
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => bail!("{name} is not valid Unicode"),
    }
}
