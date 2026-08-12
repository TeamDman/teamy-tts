use crate::backend::BackendKind;
use crate::backend::BackendSelection;
use crate::backend_receipts;
use crate::backend_receipts::BackendBenchmarkReceipt;
use crate::backend_receipts::BackendReceiptKey;
use crate::cli::output::CliOutput;
use crate::cli::say::load_runtime;
use crate::model_registry;
use arbitrary::Arbitrary;
use chrono::Utc;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::time::Instant;

const DEFAULT_VOICE: &str = "p2";
const DEFAULT_ALPHA: f32 = 1.0;
const RELATIVE_RMS_TOLERANCE: f64 = 0.20;

/// Benchmark one backend and persist a correctness-gated receipt.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct BackendBenchmarkArgs {
    /// Stable model identifier. Defaults to glados.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,

    /// Backend to benchmark: auto, burn, burn-ndarray, burn-cuda-acoustic,
    /// burn-cuda-fused, burn-tch, burn-wgpu, burn-vulkan, libtorch, or vulkan.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub backend: Option<String>,

    /// Speaker embedding to use for the benchmark corpus.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub voice: Option<String>,

    /// Duration/pitch scaling factor for the benchmark corpus.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub alpha: Option<f32>,

    /// Number of warmup corpus passes.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub warmup: Option<u32>,

    /// Number of measured corpus passes.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub measurements: Option<u32>,
}

#[derive(Clone, Debug, Facet, PartialEq)]
struct BackendBenchmarkReport {
    model: String,
    backend: String,
    device_identity: String,
    corpus: String,
    warmup_count: u32,
    measurement_count: u32,
    warm_median_ms: f64,
    warm_p95_ms: f64,
    output_sample_count: u64,
    output_duration_ms: u64,
    correctness_passed: bool,
    correctness_summary: String,
    receipt_path: String,
}

impl BackendBenchmarkArgs {
    /// # Errors
    ///
    /// Returns an error when the model/backend is unavailable, the benchmark
    /// configuration is invalid, synthesis fails, or the receipt cannot be
    /// written.
    #[expect(
        clippy::too_many_lines,
        reason = "The benchmark command keeps its load, measure, compare, and receipt sequence auditable."
    )]
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let backend_value = self.backend.as_deref();
        BackendSelection::parse(backend_value)?;
        let warmup_count = self
            .warmup
            .unwrap_or(backend_receipts::DEFAULT_WARMUP_COUNT);
        let measurement_count = self
            .measurements
            .unwrap_or(backend_receipts::DEFAULT_MEASUREMENT_COUNT);
        if warmup_count > 100 {
            bail!("--warmup must be at most 100");
        }
        if measurement_count == 0 || measurement_count > 100 {
            bail!("--measurements must be between 1 and 100");
        }
        let configuration = backend_receipts::BenchmarkConfiguration {
            corpus_id: backend_receipts::DEFAULT_BENCHMARK_CORPUS_ID.to_string(),
            warmup_count,
            measurement_count,
        };
        let corpus = configuration.corpus_texts();
        if corpus.is_empty() {
            bail!(
                "benchmark corpus {:?} is not defined",
                configuration.corpus_id
            );
        }
        let voice = self.voice.as_deref().unwrap_or(DEFAULT_VOICE);
        let alpha = self.alpha.unwrap_or(DEFAULT_ALPHA);
        let (_model, runtime) = load_runtime(model_id, backend_value)?;
        let actual_backend = runtime.backend_kind();
        tracing::info!(
            backend = %actual_backend,
            corpus = %configuration.corpus_id,
            warmup_count,
            measurement_count,
            "starting backend benchmark"
        );

        for _ in 0..warmup_count {
            for text in corpus {
                runtime.synthesize(text, voice, alpha)?;
            }
        }

        let mut elapsed_ms = Vec::with_capacity(
            usize::try_from(measurement_count).unwrap_or_default() * corpus.len(),
        );
        let mut last_outputs = Vec::with_capacity(corpus.len());
        let mut output_sample_count = 0_u64;
        for _ in 0..measurement_count {
            last_outputs.clear();
            for text in corpus {
                let started = Instant::now();
                let samples = runtime.synthesize(text, voice, alpha)?;
                let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
                    bail!("backend produced empty or non-finite audio for {text:?}");
                }
                elapsed_ms.push(elapsed);
                output_sample_count = output_sample_count
                    .checked_add(u64::try_from(samples.len()).wrap_err("sample count overflow")?)
                    .ok_or_else(|| eyre::eyre!("benchmark sample count overflowed"))?;
                last_outputs.push(samples);
            }
        }

        let (correctness_passed, correctness_summary) =
            if actual_backend == BackendKind::BurnNdArray {
                (
                    true,
                    "Burn NdArray reference output is finite and non-empty".to_string(),
                )
            } else {
                let (_model, reference) = load_runtime(model_id, Some("burn-ndarray"))?;
                let mut reference_outputs = Vec::with_capacity(corpus.len());
                for text in corpus {
                    reference_outputs.push(reference.synthesize(text, voice, alpha)?);
                }
                compare_outputs(&reference_outputs, &last_outputs)
            };

        let warm_median_ms = percentile(&mut elapsed_ms, 50);
        let warm_p95_ms = percentile(&mut elapsed_ms, 95);
        let output_duration_ms = output_sample_count
            .checked_mul(1000)
            .ok_or_else(|| eyre::eyre!("benchmark sample duration overflowed"))?
            .checked_div(u64::from(runtime.sample_rate_hz()))
            .ok_or_else(|| eyre::eyre!("sample rate cannot be zero"))?;
        let device_identity = backend_receipts::device_identity();
        let prepared = model_registry::inspect_prepared_model_dir(
            model_registry::find_model(model_id)
                .ok_or_else(|| eyre::eyre!("unknown model {model_id:?}"))?,
        )?;
        let key = BackendReceiptKey::for_model(
            &prepared,
            actual_backend,
            &device_identity,
            &configuration,
        );
        let receipt = BackendBenchmarkReceipt {
            schema_version: backend_receipts::BENCHMARK_RECEIPT_SCHEMA_VERSION,
            key,
            created_at_utc: Utc::now().to_rfc3339(),
            correctness_passed,
            correctness_summary: correctness_summary.clone(),
            warmup_count,
            measurement_count,
            warm_median_ms,
            warm_p95_ms,
            output_sample_count,
            output_duration_ms,
        };
        let receipt_path = backend_receipts::write_receipt(&receipt)?;
        tracing::info!(
            backend = %actual_backend,
            correctness_passed,
            warm_median_ms,
            receipt = %receipt_path.display(),
            "backend benchmark complete"
        );
        Ok(CliOutput::facet(BackendBenchmarkReport {
            model: model_id.to_string(),
            backend: actual_backend.to_string(),
            device_identity,
            corpus: configuration.corpus_id,
            warmup_count,
            measurement_count,
            warm_median_ms,
            warm_p95_ms,
            output_sample_count,
            output_duration_ms,
            correctness_passed,
            correctness_summary,
            receipt_path: receipt_path.display().to_string(),
        }))
    }
}

fn compare_outputs(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> (bool, String) {
    if reference.len() != candidate.len() {
        return (
            false,
            format!(
                "corpus output count differs: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ),
        );
    }
    let mut reference_energy = 0.0_f64;
    let mut difference_energy = 0.0_f64;
    let mut max_abs_error = 0.0_f64;
    for (reference, candidate) in reference.iter().zip(candidate) {
        if reference.len() != candidate.len() {
            return (
                false,
                format!(
                    "sample count differs: reference={}, candidate={}",
                    reference.len(),
                    candidate.len()
                ),
            );
        }
        for (&reference, &candidate) in reference.iter().zip(candidate) {
            if !reference.is_finite() || !candidate.is_finite() {
                return (
                    false,
                    "reference or candidate contains non-finite audio".to_string(),
                );
            }
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            reference_energy += reference * reference;
            let difference = reference - candidate;
            difference_energy += difference * difference;
            max_abs_error = max_abs_error.max(difference.abs());
        }
    }
    let relative_rms_error = if reference_energy <= f64::EPSILON {
        if difference_energy <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (difference_energy / reference_energy).sqrt()
    };
    let passed = relative_rms_error <= RELATIVE_RMS_TOLERANCE;
    (
        passed,
        format!(
            "relative_rms_error={relative_rms_error:.6};max_abs_error={max_abs_error:.6};tolerance={RELATIVE_RMS_TOLERANCE:.2}"
        ),
    )
}

fn percentile(values: &mut [f64], percentile: usize) -> f64 {
    values.sort_by(f64::total_cmp);
    let rank = values.len().saturating_mul(percentile).saturating_add(99) / 100;
    let index = rank.saturating_sub(1);
    values[index.min(values.len().saturating_sub(1))]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_outputs_pass_correctness_gate() {
        let outputs = vec![vec![0.0_f32, 0.25, -0.5]];
        assert!(compare_outputs(&outputs, &outputs).0);
    }

    #[test]
    fn materially_different_outputs_fail_correctness_gate() {
        let reference = vec![vec![1.0_f32, 1.0, 1.0]];
        let candidate = vec![vec![-1.0_f32, -1.0, -1.0]];
        assert!(!compare_outputs(&reference, &candidate).0);
    }

    #[test]
    fn percentile_uses_sorted_measurements() {
        let mut values = [30.0, 10.0, 20.0];
        assert!((percentile(&mut values, 50) - 20.0).abs() < f64::EPSILON);
    }
}
