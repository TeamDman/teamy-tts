//! Correctness and latency benchmark for the single tch/LibTorch runtime.

use crate::cli::output::CliOutput;
use crate::cli::say::load_runtime;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::time::Instant;

/// Benchmark one prepared model without writing or playing audio.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct BenchmarkArgs {
    /// Text to synthesize.
    #[facet(args::positional)]
    #[arbitrary(default)]
    pub text: String,

    /// Stable model identifier. Defaults to glados.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,

    /// Speaker embedding to use. Defaults to p2.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub voice: Option<String>,

    /// Duration/pitch scaling factor. Defaults to 1.0.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub alpha: Option<f32>,

    /// Number of additional warmup syntheses.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub warmups: Option<usize>,

    /// Number of measured syntheses.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub measurements: Option<usize>,
}

#[derive(Facet, Debug)]
struct BenchmarkReport {
    backend: String,
    model: String,
    text: String,
    model_load_ms: u128,
    warmup_count: usize,
    measurement_count: usize,
    measurement_ms: Vec<u128>,
    median_ms: u128,
    p95_ms: u128,
    sample_count: usize,
    audio_duration_ms: u128,
}

impl BenchmarkArgs {
    /// Run the configured cold-load, warmup, and measurement sequence.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let voice = self.voice.as_deref().unwrap_or("p2");
        let alpha = self.alpha.unwrap_or(1.0);
        let warmup_count = self.warmups.unwrap_or(2);
        let measurement_count = self.measurements.unwrap_or(5).max(1);

        let load_started = Instant::now();
        let (_model, runtime) = load_runtime(model_id, None)?;
        let model_load_ms = load_started.elapsed().as_millis();

        for _ in 0..warmup_count {
            let _ = runtime.synthesize(&self.text, voice, alpha)?;
        }

        let mut measurement_ms = Vec::with_capacity(measurement_count);
        let mut sample_count = 0;
        for _ in 0..measurement_count {
            let started = Instant::now();
            sample_count = runtime.synthesize(&self.text, voice, alpha)?.len();
            measurement_ms.push(started.elapsed().as_millis());
        }
        measurement_ms.sort_unstable();
        let median_ms = measurement_ms[measurement_ms.len() / 2];
        let p95_ms = measurement_ms[(measurement_ms.len() * 95).div_ceil(100).saturating_sub(1)];
        let audio_duration_ms = (sample_count as u128 * 1000) / runtime.sample_rate_hz() as u128;

        Ok(CliOutput::facet(BenchmarkReport {
            backend: runtime.backend_kind().to_string(),
            model: model_id.to_string(),
            text: self.text,
            model_load_ms,
            warmup_count,
            measurement_count,
            measurement_ms,
            median_ms,
            p95_ms,
            sample_count,
            audio_duration_ms,
        }))
    }
}
