//! Persistent benchmark evidence used by automatic backend selection.
//!
//! A receipt is deliberately tied to the prepared model, the executable's
//! backend revision, the observed device, and the benchmark configuration.
//! This prevents a fast result from one model or machine from silently
//! becoming the default for another.

use crate::backend::BackendKind;
use crate::model_registry::PreparedModelArtifacts;
use crate::paths::CACHE_DIR;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use sha2::Digest;
use sha2::Sha256;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Schema version for benchmark receipts written by this executable.
pub const BENCHMARK_RECEIPT_SCHEMA_VERSION: u32 = 2;
/// The stable benchmark corpus used for automatic backend selection.
pub const DEFAULT_BENCHMARK_CORPUS_ID: &str = "glados-short-v1";
/// Default number of warmup synthesis calls before timing.
pub const DEFAULT_WARMUP_COUNT: u32 = 2;
/// Default number of timed synthesis calls.
pub const DEFAULT_MEASUREMENT_COUNT: u32 = 3;

/// Texts in the stable short benchmark corpus.
pub static DEFAULT_BENCHMARK_TEXTS: [&str; 2] = ["Hello, friend", "Let's see how fast this goes."];

/// The long-form workload used to expose recurrent scaling and frame-count drift.
pub static LONG_BENCHMARK_TEXTS: [&str; 1] = [
    "i don't realistically see how they can keep their current approach if they want shadow mapping for vibrant visuals so i'm assuming it's something they want to do soon",
];

/// Configuration that participates in the benchmark receipt key.
#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct BenchmarkConfiguration {
    pub corpus_id: String,
    pub warmup_count: u32,
    pub measurement_count: u32,
}

impl Default for BenchmarkConfiguration {
    fn default() -> Self {
        Self {
            corpus_id: DEFAULT_BENCHMARK_CORPUS_ID.to_string(),
            warmup_count: DEFAULT_WARMUP_COUNT,
            measurement_count: DEFAULT_MEASUREMENT_COUNT,
        }
    }
}

impl BenchmarkConfiguration {
    /// Return the default workload texts for this configuration's corpus.
    #[must_use]
    pub fn corpus_texts(&self) -> &'static [&'static str] {
        match self.corpus_id.as_str() {
            DEFAULT_BENCHMARK_CORPUS_ID => &DEFAULT_BENCHMARK_TEXTS,
            "glados-long-v1" => &LONG_BENCHMARK_TEXTS,
            _ => &[],
        }
    }

    /// Return a stable human-readable configuration identity.
    #[must_use]
    pub fn stable_id(&self) -> String {
        format!(
            "corpus={};warmup={};measurements={}",
            self.corpus_id, self.warmup_count, self.measurement_count
        )
    }
}

/// The fields that must match before a receipt can influence auto.
#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct BackendReceiptKey {
    pub model_id: String,
    pub model_revision: String,
    pub prepared_model_sha256: String,
    pub backend: String,
    pub backend_revision: String,
    pub device_identity: String,
    pub benchmark_configuration: String,
}

impl BackendReceiptKey {
    /// Construct a key for one prepared model/backend/device combination.
    #[must_use]
    pub fn for_model(
        artifacts: &PreparedModelArtifacts,
        backend: BackendKind,
        device_identity: &str,
        configuration: &BenchmarkConfiguration,
    ) -> Self {
        Self {
            model_id: artifacts.manifest.model_id.clone(),
            model_revision: artifacts.manifest.revision.clone(),
            prepared_model_sha256: artifacts.manifest.source_archive_sha256.clone(),
            backend: backend.to_string(),
            backend_revision: backend_revision(backend),
            device_identity: device_identity.to_string(),
            benchmark_configuration: configuration.stable_id(),
        }
    }

    fn stable_bytes(&self) -> String {
        format!(
            "{}\0{}\0{}\0{}\0{}\0{}\0{}",
            self.model_id,
            self.model_revision,
            self.prepared_model_sha256,
            self.backend,
            self.backend_revision,
            self.device_identity,
            self.benchmark_configuration
        )
    }

    /// Return the content-addressed receipt filename component.
    #[must_use]
    pub fn digest(&self) -> String {
        let digest = Sha256::digest(self.stable_bytes().as_bytes());
        format!("{digest:x}")
    }
}

/// Measured backend evidence and the correctness result for one workload.
#[derive(Clone, Debug, Facet, PartialEq)]
pub struct BackendBenchmarkReceipt {
    pub schema_version: u32,
    pub key: BackendReceiptKey,
    pub created_at_utc: String,
    pub correctness_passed: bool,
    pub correctness_summary: String,
    pub warmup_count: u32,
    pub measurement_count: u32,
    pub warm_median_ms: f64,
    pub warm_p95_ms: f64,
    pub model_load_ms: u128,
    pub real_time_factor: f64,
    pub output_sample_count: u64,
    pub output_duration_ms: u64,
}

/// The result of automatic selection, including the receipt that justified it.
#[derive(Clone, Debug, Facet, PartialEq)]
pub struct AutoBackendDecision {
    pub backend: String,
    pub reason: String,
    pub receipt_path: Option<String>,
}

/// Select the fastest currently available passing receipt, or Burn when no
/// receipt is eligible.
#[must_use]
pub fn auto_backend_decision(
    artifacts: &PreparedModelArtifacts,
    device_identity: &str,
    configuration: &BenchmarkConfiguration,
) -> AutoBackendDecision {
    let Some((receipt, path)) = best_passing_receipt(artifacts, device_identity, configuration)
    else {
        return AutoBackendDecision {
            backend: BackendKind::Burn.to_string(),
            reason: "no passing receipt matches this model, device, backend build, and workload"
                .to_string(),
            receipt_path: None,
        };
    };

    AutoBackendDecision {
        backend: receipt.key.backend,
        reason: format!(
            "selected the fastest passing receipt at {:.3} ms median",
            receipt.warm_median_ms
        ),
        receipt_path: Some(path.display().to_string()),
    }
}

/// Return the executable revision used in backend receipt keys.
#[must_use]
pub fn backend_revision(backend: BackendKind) -> String {
    let build_variant = if cfg!(feature = "burn-cuda-fused") {
        "burn-cuda-fused-build"
    } else {
        "ordinary-build"
    };
    format!(
        "{}-{}-{}-{}-{}-{}",
        backend,
        build_variant,
        env!("CARGO_PKG_VERSION"),
        option_env!("GIT_REVISION").unwrap_or("unknown"),
        option_env!("GIT_WORKTREE_STATUS").unwrap_or("unknown"),
        option_env!("TEAMY_TTS_SOURCE_FINGERPRINT").unwrap_or("unknown")
    )
}

/// Return a stable identity for the active compute device.
///
/// `TEAMY_TTS_BENCHMARK_DEVICE_ID` is an explicit override for CI and
/// reproducible experiments. On NVIDIA systems, nvidia-smi supplies the GPU
/// UUID, name, and driver version. The fallback remains explicit as `unknown`,
/// which is still safe because it cannot match a different known identity.
#[must_use]
pub fn device_identity() -> String {
    if let Some(value) = non_empty_env("TEAMY_TTS_BENCHMARK_DEVICE_ID") {
        return value;
    }

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();
    if let Ok(output) = output
        && output.status.success()
        && let Ok(value) = String::from_utf8(output.stdout)
    {
        let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
        if !value.is_empty() {
            return value;
        }
    }

    crate::config::effective_torch_device()
        .ok()
        .flatten()
        .map_or_else(|| "unknown".to_string(), |device| format!("cuda:{device}"))
}

/// Return the root directory containing benchmark receipts for one model.
#[must_use]
pub fn receipt_directory(model_id: &str) -> PathBuf {
    CACHE_DIR.0.join("backend-benchmarks").join(model_id)
}

/// Return the path for a receipt key.
#[must_use]
pub fn receipt_path(key: &BackendReceiptKey) -> PathBuf {
    receipt_directory(&key.model_id).join(format!("{}.json", key.digest()))
}

/// Atomically persist a benchmark receipt.
///
/// # Errors
///
/// Returns an error when the receipt cannot be serialized or written.
pub fn write_receipt(receipt: &BackendBenchmarkReceipt) -> Result<PathBuf> {
    let path = receipt_path(&receipt.key);
    let parent = path
        .parent()
        .ok_or_else(|| eyre::eyre!("benchmark receipt has no parent: {}", path.display()))?;
    fs::create_dir_all(parent).wrap_err_with(|| {
        format!(
            "failed to create benchmark receipt directory {}",
            parent.display()
        )
    })?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("receipt.json");
    let temporary = path.with_file_name(format!("{filename}.partial-{}", std::process::id()));
    let contents =
        facet_json::to_string_pretty(receipt).wrap_err("failed to serialize benchmark receipt")?;
    fs::write(&temporary, contents)
        .wrap_err_with(|| format!("failed to write benchmark receipt {}", temporary.display()))?;
    if path.exists() {
        fs::remove_file(&path)
            .wrap_err_with(|| format!("failed to replace benchmark receipt {}", path.display()))?;
    }
    fs::rename(&temporary, &path)
        .wrap_err_with(|| format!("failed to install benchmark receipt {}", path.display()))?;
    Ok(path)
}

/// Load the best passing receipt matching the current model/device/workload.
///
/// Malformed, stale, or failing receipts are ignored. A receipt only becomes
/// eligible after the backend reports it as currently available.
#[must_use]
pub fn best_passing_receipt(
    artifacts: &PreparedModelArtifacts,
    device_identity: &str,
    configuration: &BenchmarkConfiguration,
) -> Option<(BackendBenchmarkReceipt, PathBuf)> {
    let directory = receipt_directory(&artifacts.manifest.model_id);
    let entries = fs::read_dir(directory).ok()?;
    let mut best: Option<(BackendBenchmarkReceipt, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(receipt) = facet_json::from_str::<BackendBenchmarkReceipt>(&contents) else {
            continue;
        };
        let Ok(backend) = BackendKind::parse(&receipt.key.backend) else {
            continue;
        };
        if receipt.schema_version != BENCHMARK_RECEIPT_SCHEMA_VERSION
            || !receipt.correctness_passed
            || receipt.key.model_id != artifacts.manifest.model_id
            || receipt.key.model_revision != artifacts.manifest.revision
            || receipt.key.prepared_model_sha256 != artifacts.manifest.source_archive_sha256
            || receipt.key.device_identity != device_identity
            || receipt.key.benchmark_configuration != configuration.stable_id()
            || receipt.key.backend_revision != backend_revision(backend)
            || !receipt.warm_median_ms.is_finite()
            || !backend_is_available(&receipt.key.backend)
        {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(current, _)| receipt.warm_median_ms < current.warm_median_ms)
        {
            best = Some((receipt, path));
        }
    }
    best
}

/// Check whether a concrete backend can be loaded in this build/environment.
#[must_use]
pub fn backend_is_available(backend: &str) -> bool {
    match BackendKind::parse(backend) {
        Ok(BackendKind::Burn | BackendKind::BurnNdArray) => true,
        Ok(BackendKind::BurnCudaAcoustic) => {
            cfg!(feature = "cuda")
        }
        Ok(BackendKind::BurnCudaFused) => {
            cfg!(feature = "burn-cuda-fused")
        }
        Ok(BackendKind::BurnTch) => {
            cfg!(feature = "burn-tch")
        }
        Ok(BackendKind::BurnWgpu) => {
            cfg!(feature = "burn-wgpu")
        }
        Ok(BackendKind::BurnVulkan) => cfg!(feature = "burn-vulkan"),
        Err(_) => false,
        Ok(BackendKind::LibTorch) => {
            #[cfg(feature = "torchscript")]
            {
                let Ok(Some(model_dir)) = crate::config::effective_torch_model_dir() else {
                    return false;
                };
                model_dir.join("glados-new.pt").is_file()
                    && model_dir.join("vocoder-gpu.pt").is_file()
            }
            #[cfg(not(feature = "torchscript"))]
            {
                false
            }
        }
        Ok(BackendKind::Vulkan) => {
            #[cfg(feature = "vulkan")]
            {
                true
            }
            #[cfg(not(feature = "vulkan"))]
            {
                false
            }
        }
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_id_is_stable() {
        assert_eq!(
            BenchmarkConfiguration::default().stable_id(),
            "corpus=glados-short-v1;warmup=2;measurements=3"
        );
    }

    #[test]
    fn receipt_key_digest_changes_when_device_changes() {
        let left = BackendReceiptKey {
            model_id: "glados".to_string(),
            model_revision: "glados-new".to_string(),
            prepared_model_sha256: "model-hash".to_string(),
            backend: "burn".to_string(),
            backend_revision: "burn-rev".to_string(),
            device_identity: "device-a".to_string(),
            benchmark_configuration: "config".to_string(),
        };
        let mut right = left.clone();
        right.device_identity = "device-b".to_string();
        assert_ne!(left.digest(), right.digest());
    }

    #[test]
    fn unavailable_backend_is_not_eligible() {
        #[cfg(feature = "vulkan")]
        assert!(backend_is_available("vulkan"));
        #[cfg(not(feature = "vulkan"))]
        assert!(!backend_is_available("vulkan"));
        assert!(!backend_is_available("not-a-backend"));
    }
}
