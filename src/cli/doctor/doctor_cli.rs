//! Non-mutating diagnostics for the local model and native runtime.

use crate::cli::output::CliOutput;
use crate::config;
use crate::model_registry;
use crate::model_sources;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

const REPORT_SCHEMA_VERSION: u32 = 1;
const DEFAULT_MODEL_ID: &str = "glados";
const DOCTOR_TEXT: &str = "Hello, friend";

/// The outcome of one diagnostic check or the aggregate report.
#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "kebab-case")]
#[repr(u8)]
pub enum DoctorStatus {
    Pass,
    Warn,
    Fail,
    Skip,
}

/// One safe, machine-readable diagnostic result.
#[derive(Clone, Debug, Facet, PartialEq)]
#[facet(rename_all = "kebab-case")]
pub struct DoctorCheck {
    pub id: String,
    pub status: DoctorStatus,
    pub summary: String,
    pub evidence: Option<String>,
    pub remediation: Option<String>,
}

/// The complete non-mutating health report.
#[derive(Clone, Debug, Facet, PartialEq)]
#[facet(rename_all = "kebab-case")]
pub struct DoctorReport {
    pub schema_version: u32,
    pub status: DoctorStatus,
    pub model: String,
    pub deep: bool,
    pub offline: bool,
    pub checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    fn csv_projection(&self) -> String {
        let mut csv = String::from(
            "schema-version,status,model,deep,offline,check-id,check-status,summary,evidence,remediation\n",
        );
        for check in &self.checks {
            let fields = [
                self.schema_version.to_string(),
                status_label(self.status).to_string(),
                self.model.clone(),
                self.deep.to_string(),
                self.offline.to_string(),
                check.id.clone(),
                status_label(check.status).to_string(),
                check.summary.clone(),
                check.evidence.clone().unwrap_or_default(),
                check.remediation.clone().unwrap_or_default(),
            ];
            csv.push_str(
                &fields
                    .iter()
                    .map(|field| csv_field(field))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            csv.push('\n');
        }
        csv
    }
}

/// Inspect model, configuration, native runtime, CUDA, audio, and public
/// model-server prerequisites.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct DoctorArgs {
    /// Stable model identifier. Defaults to `glados`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,

    /// Load the actual models and perform a short in-memory synthesis smoke test.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub deep: bool,

    /// Skip network probes for model source endpoints.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub offline: bool,
}

impl DoctorArgs {
    /// Run all requested diagnostics without changing configuration, cache, or
    /// model files.
    ///
    /// # Errors
    ///
    /// Returns an error only when the typed report cannot be constructed or a
    /// required diagnostic dependency cannot be initialized. Individual health
    /// failures are represented in the report.
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or(DEFAULT_MODEL_ID);
        let mut checks = Vec::new();
        let effective_config = check_configuration(&mut checks);
        let model = model_registry::find_model(model_id);
        check_model_catalog(model_id, model, &mut checks);

        let prepared = if let Some(model) = model {
            check_prepared_model(model, self.deep, &mut checks)
        } else {
            None
        };
        check_torch_model_dir(effective_config.as_ref(), &mut checks);
        check_output_directory(&mut checks);
        check_audio_support(&mut checks);
        check_native_runtime(effective_config.as_ref(), &mut checks);

        if self.deep {
            check_deep_runtime(model_id, model, prepared.is_some(), &mut checks);
        } else {
            checks.push(DoctorCheck::skip(
                "runtime.deep-synthesis",
                "Deep model loading and synthesis were not requested",
                "Run `teamy-tts doctor --deep` to exercise the actual inference path",
            ));
        }

        checks.extend(check_model_servers(model, self.offline).await);

        let status = aggregate_status(&checks);
        let report = DoctorReport {
            schema_version: REPORT_SCHEMA_VERSION,
            status,
            model: model_id.to_string(),
            deep: self.deep,
            offline: self.offline,
            checks,
        };
        Ok(CliOutput::facet_with_csv(
            report.clone(),
            report.csv_projection(),
        ))
    }
}

fn check_configuration(checks: &mut Vec<DoctorCheck>) -> Option<config::EffectiveConfig> {
    match config::effective() {
        Ok(effective) => {
            let torch_model_dir = effective
                .torch_model_dir
                .as_deref()
                .unwrap_or("<not configured>");
            checks.push(DoctorCheck::pass(
                "configuration.effective",
                "Effective configuration is readable",
                format!(
                    "backend={}, model-dir={}, torch-model-dir={}, torch-device={}",
                    effective.backend,
                    effective.model_dir,
                    torch_model_dir,
                    effective
                        .torch_device
                        .map_or_else(|| "default".to_string(), |device| device.to_string())
                ),
            ));
            match config::load() {
                Ok(stored) => checks.push(DoctorCheck::pass(
                    "configuration.precedence",
                    "Configuration precedence is inspectable without exposing values",
                    format!(
                        "backend={}, model-dir={}, torch-model-dir={}, torch-device={}",
                        config_source(
                            config::BACKEND_ENV_VAR,
                            stored.backend.is_some(),
                            "default-auto"
                        ),
                        config_source(
                            config::MODEL_DIR_ENV_VAR,
                            stored.model_dir.is_some(),
                            "default-cache"
                        ),
                        config_source(
                            config::TORCH_MODEL_DIR_ENV_VAR,
                            stored.torch_model_dir.is_some(),
                            "unconfigured"
                        ),
                        config_source(
                            config::TORCH_DEVICE_ENV_VAR,
                            stored.torch_device.is_some(),
                            "default-device-0"
                        )
                    ),
                )),
                Err(error) => checks.push(DoctorCheck::fail(
                    "configuration.precedence",
                    "Remembered configuration could not be inspected",
                    error.to_string(),
                    "Run `teamy-tts config show` after correcting the configuration file",
                )),
            }
            Some(effective)
        }
        Err(error) => {
            checks.push(DoctorCheck::fail(
                "configuration.effective",
                "Effective configuration could not be resolved",
                error.to_string(),
                "Run `teamy-tts config show` and correct the reported configuration value",
            ));
            None
        }
    }
}

fn config_source(environment_variable: &str, stored: bool, default: &str) -> String {
    if std::env::var_os(environment_variable).is_some() {
        format!("environment-override:{environment_variable}")
    } else if stored {
        "remembered-config".to_string()
    } else {
        default.to_string()
    }
}

fn check_model_catalog(
    model_id: &str,
    model: Option<model_registry::ModelDefinition>,
    checks: &mut Vec<DoctorCheck>,
) {
    match model {
        Some(model) => checks.push(DoctorCheck::pass(
            "model.catalog",
            format!("Model {model_id:?} is known to the catalog"),
            format!(
                "revision={}, sample-rate-hz={}",
                model.revision, model.sample_rate_hz
            ),
        )),
        None => checks.push(DoctorCheck::fail(
            "model.catalog",
            format!("Model {model_id:?} is not known to the catalog"),
            "known-models=glados",
            "Use `--model glados`",
        )),
    }
}

fn check_prepared_model(
    model: model_registry::ModelDefinition,
    deep: bool,
    checks: &mut Vec<DoctorCheck>,
) -> Option<model_registry::PreparedModelArtifacts> {
    match model_registry::inspect_prepared_model_dir(model) {
        Ok(artifacts) => {
            if deep {
                match model_registry::verify_prepared_model_artifacts(&artifacts) {
                    Ok(()) => checks.push(DoctorCheck::pass(
                        "model.prepared",
                        "Prepared model manifest and artifact hashes are valid",
                        format!(
                            "root={}, artifacts={}, verification=sha256",
                            artifacts.root.display(),
                            artifacts.manifest.artifacts.len()
                        ),
                    )),
                    Err(error) => {
                        checks.push(DoctorCheck::fail(
                            "model.prepared",
                            "Prepared model files do not match their manifest hashes",
                            error.to_string(),
                            "Run `teamy-tts model prepare glados --force` or reacquire the prepared bundle",
                        ));
                        return None;
                    }
                }
            } else {
                checks.push(DoctorCheck::pass(
                    "model.prepared",
                    "Prepared model manifest and artifact sizes are valid",
                    format!(
                        "root={}, artifacts={}, verification=size-and-manifest",
                        artifacts.root.display(),
                        artifacts.manifest.artifacts.len()
                    ),
                ));
            }
            Some(artifacts)
        }
        Err(error) => {
            checks.push(DoctorCheck::fail(
                "model.prepared",
                "Prepared model is unavailable or invalid",
                error.to_string(),
                "Run `teamy-tts model acquire-prepared Teamy` or `teamy-tts model prepare glados`",
            ));
            None
        }
    }
}

fn check_torch_model_dir(
    effective_config: Option<&config::EffectiveConfig>,
    checks: &mut Vec<DoctorCheck>,
) {
    let Some(effective_config) = effective_config else {
        checks.push(DoctorCheck::skip(
            "model.torch-source",
            "The TorchScript source directory could not be evaluated because configuration failed",
            "Fix the configuration check first",
        ));
        return;
    };
    let Some(path) = effective_config.torch_model_dir.as_deref() else {
        checks.push(DoctorCheck::fail(
            "model.torch-source",
            "The upstream TorchScript model directory is not configured",
            "torch-model-dir=<not configured>",
            "Run `teamy-tts config set --torch-model-dir <path>` once",
        ));
        return;
    };

    let path = Path::new(path);
    if !path.is_dir() {
        checks.push(DoctorCheck::fail(
            "model.torch-source",
            "The configured TorchScript model directory is missing",
            format!("path={}", path.display()),
            "Set the path to the upstream model directory with `teamy-tts config set --torch-model-dir <path>`",
        ));
        return;
    }

    let required = ["glados-new.pt", "vocoder-gpu.pt"];
    let missing = required
        .iter()
        .filter(|file| !path.join(file).is_file())
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        checks.push(DoctorCheck::pass(
            "model.torch-source",
            "The configured TorchScript source directory contains the required upstream graphs",
            format!("path={}", path.display()),
        ));
    } else {
        checks.push(DoctorCheck::fail(
            "model.torch-source",
            "The configured TorchScript source directory is incomplete",
            format!("path={}, missing={missing:?}", path.display()),
            "Point `--torch-model-dir` at the upstream models directory containing glados-new.pt and vocoder-gpu.pt",
        ));
    }
}

fn check_output_directory(checks: &mut Vec<DoctorCheck>) {
    let output_dir = PathBuf::from("outputs");
    match std::fs::metadata(&output_dir) {
        Ok(metadata) if metadata.is_dir() && !metadata.permissions().readonly() => {
            checks.push(DoctorCheck::pass(
                "audio.output-directory",
                "The optional output directory exists and is not marked read-only",
                format!(
                    "path={}, writability=not-marked-read-only; no write probe performed",
                    output_dir.display()
                ),
            ));
        }
        Ok(metadata) if metadata.is_dir() => checks.push(DoctorCheck::fail(
            "audio.output-directory",
            "The optional output directory is marked read-only",
            format!("path={}", output_dir.display()),
            "Choose a writable location with --output-dir",
        )),
        Ok(_) => checks.push(DoctorCheck::fail(
            "audio.output-directory",
            "The optional output path exists but is not a directory",
            format!("path={}", output_dir.display()),
            "Remove or rename the conflicting outputs path",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            checks.push(DoctorCheck::pass(
                "audio.output-directory",
                "No default output directory is required; persistent WAV output is opt-in",
                format!("path={}; not-created", output_dir.display()),
            ));
        }
        Err(error) => checks.push(DoctorCheck::fail(
            "audio.output-directory",
            "The optional output directory could not be inspected",
            format!("path={}, error={error}", output_dir.display()),
            "Choose a writable location with --output-dir",
        )),
    }
}

fn check_audio_support(checks: &mut Vec<DoctorCheck>) {
    if cfg!(windows) {
        #[cfg(windows)]
        {
            // SAFETY: `waveOutGetNumDevs` is a read-only WinMM capability
            // query and has no pointer or lifetime arguments.
            let device_count = unsafe { windows::Win32::Media::Audio::waveOutGetNumDevs() };
            if device_count == 0 {
                checks.push(DoctorCheck::warn(
                    "audio.playback",
                    "Windows WAV playback is compiled in but no wave output device was reported",
                    "adapter=Win32 PlaySoundW; wave-output-devices=0",
                    "Connect or enable a Windows audio output device, or use `write`",
                ));
            } else {
                checks.push(DoctorCheck::pass(
                    "audio.playback",
                    "Windows WAV playback support and an output device are available",
                    format!(
                        "adapter=Win32 PlaySoundW/SND_MEMORY; wave-output-devices={device_count}"
                    ),
                ));
            }
        }
    } else {
        checks.push(DoctorCheck::warn(
            "audio.playback",
            "Synchronous playback is unavailable on this platform",
            "adapter=unsupported; WAV writing remains available",
            "Use `write` and play the WAV with another local audio tool",
        ));
    }
}

fn check_native_runtime(
    effective_config: Option<&config::EffectiveConfig>,
    checks: &mut Vec<DoctorCheck>,
) {
    #[cfg(feature = "tch-native")]
    {
        let has_cuda = tch::utils::has_cuda();
        let has_cudart = tch::utils::has_cudart();
        let cudnn = tch::Cuda::cudnn_is_available();
        let device_count = tch::Cuda::device_count();
        let cudart_version = tch::utils::version_cudart();
        let cudnn_version = tch::utils::version_cudnn();
        checks.push(DoctorCheck::pass(
            "runtime.libtorch",
            "The tch/LibTorch native runtime loaded into this process",
            "backend=tch; torch-sys=0.24.0; libtorch=2.11.x",
        ));
        let requested_device = effective_config.and_then(|config| config.torch_device);
        if requested_device.is_some_and(|device| device < 0) {
            checks.push(DoctorCheck::pass(
                "runtime.cuda",
                "CUDA was intentionally disabled by the configured Torch device",
                format!(
                    "device-count={device_count}, cudart={has_cudart}, cudart-version={cudart_version}, cudnn={cudnn}, cudnn-version={cudnn_version}"
                ),
            ));
        } else if has_cuda && has_cudart && device_count > 0 {
            checks.push(DoctorCheck::pass(
                "runtime.cuda",
                "LibTorch reports CUDA devices available",
                format!(
                    "device-count={device_count}, cudart={has_cudart}, cudart-version={cudart_version}, cudnn={cudnn}, cudnn-version={cudnn_version}"
                ),
            ));
        } else {
            checks.push(DoctorCheck::fail(
                "runtime.cuda",
                "The configured CUDA inference path is unavailable",
                format!(
                    "has-cuda={has_cuda}, cudart={has_cudart}, cudart-version={cudart_version}, cudnn={cudnn}, cudnn-version={cudnn_version}, device-count={device_count}"
                ),
                "Install the matching NVIDIA runtime or set `teamy-tts config set --torch-device -1` for CPU mode",
            ));
        }
    }
    #[cfg(not(feature = "tch-native"))]
    {
        let _ = effective_config;
        checks.push(DoctorCheck::skip(
            "runtime.libtorch",
            "This build was compiled without the tch-native feature",
            "Rebuild with the default tch-native feature",
        ));
        checks.push(DoctorCheck::skip(
            "runtime.cuda",
            "CUDA diagnostics require the tch-native feature",
            "Rebuild with the default tch-native feature",
        ));
    }
}

fn check_deep_runtime(
    model_id: &str,
    model: Option<model_registry::ModelDefinition>,
    prepared: bool,
    checks: &mut Vec<DoctorCheck>,
) {
    if model.is_none() || !prepared {
        checks.push(DoctorCheck::skip(
            "runtime.deep-synthesis",
            "Deep synthesis was skipped because the catalog or prepared model check failed",
            "Resolve the earlier model checks, then rerun with --deep",
        ));
        return;
    }
    match crate::cli::say::load_runtime(model_id, None)
        .and_then(|(_, runtime)| runtime.synthesize(DOCTOR_TEXT, "p2", 1.0))
    {
        Ok(samples) if !samples.is_empty() && samples.iter().all(|sample| sample.is_finite()) => {
            checks.push(DoctorCheck::pass(
                "runtime.deep-synthesis",
                "The loaded model completed an in-memory synthesis smoke test",
                format!("sample-count={}", samples.len()),
            ));
        }
        Ok(_) => checks.push(DoctorCheck::fail(
            "runtime.deep-synthesis",
            "The synthesis smoke test returned no usable audio samples",
            "samples=empty-or-non-finite",
            "Inspect the native model files and LibTorch/CUDA checks, then rerun with --deep",
        )),
        Err(error) => checks.push(DoctorCheck::fail(
            "runtime.deep-synthesis",
            "The loaded model failed the synthesis smoke test",
            error.to_string(),
            "Inspect the earlier model, TorchScript, and CUDA checks",
        )),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "The endpoint probe keeps URL safety, timeout, status, and size validation together."
)]
async fn check_model_servers(
    model: Option<model_registry::ModelDefinition>,
    offline: bool,
) -> Vec<DoctorCheck> {
    if offline {
        return vec![DoctorCheck::skip(
            "model-server.endpoints",
            "Model-server probes were skipped by --offline",
            "Rerun without --offline to probe configured public endpoints",
        )];
    }
    let Some(model) = model else {
        return vec![DoctorCheck::skip(
            "model-server.endpoints",
            "Model-server probes were skipped because the model is unknown",
            "Use a known model identifier",
        )];
    };

    let endpoints = match model_sources::diagnostic_source_endpoints() {
        Ok(endpoints) => endpoints,
        Err(error) => {
            return vec![DoctorCheck::fail(
                "model-server.endpoints",
                "Configured model-server endpoints could not be resolved",
                error.to_string(),
                "Unset the invalid model-source URL override or provide a valid HTTPS URL",
            )];
        }
    };
    if endpoints.is_empty() {
        return vec![DoctorCheck::warn(
            "model-server.endpoints",
            "No model-server endpoints are configured",
            "endpoint-count=0",
            "Configure a public model source before attempting acquisition",
        )];
    }

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return vec![DoctorCheck::fail(
                "model-server.endpoints",
                "The HTTP client for model-server probes could not be created",
                error.to_string(),
                "Check the application TLS/runtime installation",
            )];
        }
    };

    let mut checks = Vec::with_capacity(endpoints.len());
    for endpoint in endpoints {
        let expected_size = match endpoint.artifact {
            "raw-model-archive" => model.archive_size_bytes,
            "native-bundle" => model.native_bundle_size_bytes,
            _ => 0,
        };
        let check_id = format!(
            "model-server.{}.{}",
            endpoint.source.to_ascii_lowercase(),
            endpoint.artifact
        );
        let Some(endpoint_url) = endpoint.url else {
            checks.push(DoctorCheck::skip(
                check_id,
                "No public URL is configured for this model source",
                "Configure the source URL environment override before probing it",
            ));
            continue;
        };
        let parsed = match reqwest::Url::parse(&endpoint_url) {
            Ok(url) if url.scheme() == "https" => url,
            Ok(url) => {
                checks.push(DoctorCheck::fail(
                    check_id,
                    "Model-server endpoint is not HTTPS",
                    format!("scheme={}", url.scheme()),
                    "Use an HTTPS model-server URL",
                ));
                continue;
            }
            Err(_) => {
                checks.push(DoctorCheck::fail(
                    check_id,
                    "Model-server endpoint is not a valid URL",
                    "url=invalid",
                    "Set the source URL to a valid HTTPS URL",
                ));
                continue;
            }
        };

        match client.head(parsed).send().await {
            Ok(response) if response.status().is_success() => {
                let content_length = response.content_length();
                if content_length.is_some_and(|size| size != expected_size) {
                    checks.push(DoctorCheck::fail(
                        check_id,
                        "Model-server endpoint responded with an unexpected artifact size",
                        format!(
                            "expected-bytes={expected_size}, content-length={content_length:?}"
                        ),
                        "Verify the published object and catalog revision",
                    ));
                } else {
                    checks.push(DoctorCheck::pass(
                        check_id,
                        "Model-server endpoint responded successfully",
                        format!("source={}, expected-bytes={expected_size}", endpoint.source),
                    ));
                }
            }
            Ok(response) => checks.push(DoctorCheck::fail(
                check_id,
                "Model-server endpoint returned an unsuccessful HTTP status",
                format!("status={}", response.status()),
                "Verify the public object URL and model publication",
            )),
            Err(_) => checks.push(DoctorCheck::fail(
                check_id,
                "Model-server endpoint could not be reached within the diagnostic timeout",
                format!("source={}, timeout-seconds=5", endpoint.source),
                "Check network access or rerun with --offline when working without a network",
            )),
        }
    }
    checks
}

fn aggregate_status(checks: &[DoctorCheck]) -> DoctorStatus {
    if checks
        .iter()
        .any(|check| check.status == DoctorStatus::Fail)
    {
        DoctorStatus::Fail
    } else if checks
        .iter()
        .any(|check| check.status == DoctorStatus::Warn)
    {
        DoctorStatus::Warn
    } else if checks
        .iter()
        .all(|check| check.status == DoctorStatus::Skip)
    {
        DoctorStatus::Skip
    } else {
        DoctorStatus::Pass
    }
}

fn status_label(status: DoctorStatus) -> &'static str {
    match status {
        DoctorStatus::Pass => "pass",
        DoctorStatus::Warn => "warn",
        DoctorStatus::Fail => "fail",
        DoctorStatus::Skip => "skip",
    }
}

fn csv_field(value: &str) -> String {
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\r' | '\n'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

impl DoctorCheck {
    fn pass(
        id: impl Into<String>,
        summary: impl Into<String>,
        evidence: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            status: DoctorStatus::Pass,
            summary: summary.into(),
            evidence: Some(evidence.into()),
            remediation: None,
        }
    }

    fn warn(
        id: impl Into<String>,
        summary: impl Into<String>,
        evidence: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            status: DoctorStatus::Warn,
            summary: summary.into(),
            evidence: Some(evidence.into()),
            remediation: Some(remediation.into()),
        }
    }

    fn fail(
        id: impl Into<String>,
        summary: impl Into<String>,
        evidence: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            status: DoctorStatus::Fail,
            summary: summary.into(),
            evidence: Some(evidence.into()),
            remediation: Some(remediation.into()),
        }
    }

    fn skip(
        id: impl Into<String>,
        summary: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            status: DoctorStatus::Skip,
            summary: summary.into(),
            evidence: None,
            remediation: Some(remediation.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_status_prioritizes_failures_then_warnings() {
        let checks = vec![
            DoctorCheck::pass("one", "ok", "evidence"),
            DoctorCheck::warn("two", "warning", "evidence", "fix"),
        ];
        assert_eq!(aggregate_status(&checks), DoctorStatus::Warn);

        let mut checks = checks;
        checks.push(DoctorCheck::fail("three", "failure", "evidence", "fix"));
        assert_eq!(aggregate_status(&checks), DoctorStatus::Fail);
    }

    #[test]
    fn aggregate_status_is_skip_when_every_check_is_skipped() {
        let checks = vec![DoctorCheck::skip("one", "skipped", "rerun")];
        assert_eq!(aggregate_status(&checks), DoctorStatus::Skip);
    }

    #[test]
    fn report_does_not_have_a_secret_value_field() {
        let report = DoctorReport {
            schema_version: REPORT_SCHEMA_VERSION,
            status: DoctorStatus::Pass,
            model: DEFAULT_MODEL_ID.to_string(),
            deep: false,
            offline: true,
            checks: vec![DoctorCheck::pass(
                "config.secret",
                "A secret is configured",
                "configured=true; source=environment",
            )],
        };
        let json = facet_json::to_string(&report).expect("doctor report should serialize");
        assert!(json.contains("configured=true"));
        assert!(!json.contains("token"));
        assert!(!json.contains("secret-value"));
    }

    #[test]
    fn all_report_formats_keep_the_safe_projection_secret_free() {
        let report = DoctorReport {
            schema_version: REPORT_SCHEMA_VERSION,
            status: DoctorStatus::Pass,
            model: DEFAULT_MODEL_ID.to_string(),
            deep: false,
            offline: true,
            checks: vec![DoctorCheck::pass(
                "configuration.secret-presence",
                "A secret is configured",
                "configured=true; source=environment",
            )],
        };
        let text = facet_pretty::PrettyPrinter::new()
            .with_colors(facet_pretty::ColorMode::Never)
            .format(&report);
        let json = facet_json::to_string_pretty(&report).expect("JSON should serialize");
        let csv = report.csv_projection();
        for rendered in [&text, &json, &csv] {
            assert!(rendered.contains("configuration.secret-presence"));
            assert!(!rendered.contains("super-secret-token"));
        }
    }
}
