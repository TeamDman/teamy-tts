//! End-to-end `tch`/`LibTorch` `GLaDOS` inference and WAV emission.

use crate::frontend::GladosFrontend;
use crate::frontend_model::GladosPhonemizer;
use crate::model_registry::PreparedModelArtifacts;
use crate::native_glados::tch_torchscript::TchTorchScriptRuntime;
use eyre::Context;
use eyre::bail;
use std::path::Path;
use std::time::Instant;

/// Runtime artifact roles required for local `GLaDOS` synthesis.
pub const FRONTEND_ROLE: &str = "frontend-dictionary";
pub const VOICE_P1_ROLE: &str = "voice-p1";
pub const VOICE_P2_ROLE: &str = "voice-p2";

/// A loaded single-backend `GLaDOS` inference pipeline.
#[derive(Debug)]
pub struct GladosRuntime {
    engine: TchTorchScriptRuntime,
    frontend: GladosTextFrontend,
    sample_rate_hz: u32,
}

/// The text-side portion of the `GLaDOS` frontend, without the acoustic model
/// or vocoder. This is enough for inspecting or displaying phonemes.
#[derive(Debug)]
pub struct GladosTextFrontend {
    frontend: GladosFrontend,
    phonemizer: GladosPhonemizer,
}

impl GladosTextFrontend {
    /// Load the prepared dictionary and the frontend phonemizer checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the prepared dictionary or phonemizer checkpoint
    /// cannot be loaded.
    pub fn from_prepared(
        artifacts: &PreparedModelArtifacts,
        model_dir: &Path,
    ) -> eyre::Result<Self> {
        let frontend_path = artifacts.path_for_role(FRONTEND_ROLE)?;
        let bundled_phonemizer_path = artifacts.root.join("glados-phonemizer.pt");
        let phonemizer_path = if bundled_phonemizer_path.is_file() {
            bundled_phonemizer_path
        } else {
            model_dir.join("glados-phonemizer.pt")
        };
        if !phonemizer_path.is_file() {
            bail!(
                "tch phonemizer model is missing: {}; export it with tools/export_glados_phonemizer.py",
                phonemizer_path.display()
            );
        }

        tracing::info!(path = %phonemizer_path.display(), "loading phonemizer model through tch");
        let phonemizer = GladosPhonemizer::from_file(&phonemizer_path)?;
        tracing::info!("loading frontend dictionary");
        let frontend = GladosFrontend::from_tsv(frontend_path)?;
        Ok(Self {
            frontend,
            phonemizer,
        })
    }

    /// Return the normalized and phonemized `GLaDOS` symbol sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when neural phonemization fails or emits an unsupported
    /// symbol.
    pub fn phonemize(&self, text: &str) -> eyre::Result<String> {
        self.frontend
            .phonemize_with(text, |word| self.phonemizer.phonemize_word(word))
    }

    /// Normalize, phonemize, and convert ordinary text to model token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error when neural phonemization fails or emits an unsupported
    /// symbol.
    pub fn tokenize_text(&self, text: &str) -> eyre::Result<Vec<i32>> {
        self.frontend
            .tokenize_with(text, |word| self.phonemizer.phonemize_word(word))
    }

    /// Convert a phoneme-symbol sequence to the model's integer token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error when the sequence contains an unsupported symbol.
    pub fn tokenize_phonemes(&self, phonemes: &str) -> eyre::Result<Vec<i32>> {
        self.frontend.tokenize_phonemes(phonemes)
    }
}

impl GladosRuntime {
    /// Load the prepared frontend artifacts and raw upstream `TorchScript` model.
    ///
    /// # Errors
    ///
    /// Returns an error when prepared artifacts, the phonemizer, `TorchScript`
    /// graphs, or the configured native runtime cannot be loaded.
    pub fn from_prepared(
        artifacts: &PreparedModelArtifacts,
        model_dir: &Path,
    ) -> eyre::Result<Self> {
        let voice_p1_path = artifacts.path_for_role(VOICE_P1_ROLE)?;
        let voice_p2_path = artifacts.path_for_role(VOICE_P2_ROLE)?;
        let frontend = GladosTextFrontend::from_prepared(artifacts, model_dir)?;
        let engine =
            TchTorchScriptRuntime::from_model_dir(model_dir, voice_p1_path, voice_p2_path)?;
        Ok(Self {
            engine,
            frontend,
            sample_rate_hz: artifacts.manifest.sample_rate_hz,
        })
    }

    /// Generate mono floating-point audio for one English utterance.
    ///
    /// # Errors
    ///
    /// Returns an error when the input parameters are invalid, phonemization
    /// fails, or native inference fails.
    pub fn synthesize(&self, text: &str, voice: &str, alpha: f32) -> eyre::Result<Vec<f32>> {
        if !alpha.is_finite() || alpha <= 0.0 {
            bail!("alpha must be a finite positive number");
        }

        tracing::info!("phonemizing input");
        let token_values = self.frontend.tokenize_text(text)?;
        tracing::info!(token_count = token_values.len(), "phonemization complete");
        self.synthesize_tokens(&token_values, voice, alpha)
    }

    /// Generate mono floating-point audio from `GLaDOS` IPA-like phoneme
    /// symbols, bypassing English normalization and the neural phonemizer.
    ///
    /// # Errors
    ///
    /// Returns an error when the input contains an unsupported symbol or
    /// native inference fails.
    pub fn synthesize_phonemes(
        &self,
        phonemes: &str,
        voice: &str,
        alpha: f32,
    ) -> eyre::Result<Vec<f32>> {
        if !alpha.is_finite() || alpha <= 0.0 {
            bail!("alpha must be a finite positive number");
        }

        tracing::info!("validating phoneme input");
        let token_values = self.frontend.tokenize_phonemes(phonemes)?;
        tracing::info!(token_count = token_values.len(), "phoneme input validated");
        self.synthesize_tokens(&token_values, voice, alpha)
    }

    fn synthesize_tokens(
        &self,
        token_values: &[i32],
        voice: &str,
        alpha: f32,
    ) -> eyre::Result<Vec<f32>> {
        tracing::info!("synthesizing with tch/LibTorch");
        let started = Instant::now();
        let samples = self.engine.synthesize(token_values, voice, alpha)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            sample_count = samples.len(),
            "tch/LibTorch synthesis complete"
        );
        Ok(samples)
    }

    /// Write mono floating-point samples as a 16-bit PCM WAV file.
    ///
    /// # Errors
    ///
    /// Returns an error when the samples cannot be represented by a WAV file
    /// or the destination cannot be written.
    pub fn write_wav(&self, output: &Path, samples: &[f32]) -> eyre::Result<()> {
        let wav = self.wav_bytes(samples)?;
        self.write_wav_bytes(output, &wav)
    }

    /// Encode mono floating-point samples as a 16-bit PCM WAV buffer in
    /// memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the samples cannot be represented by a WAV file.
    pub fn wav_bytes(&self, samples: &[f32]) -> eyre::Result<Vec<u8>> {
        encode_pcm16_wav(self.sample_rate_hz, samples)
    }

    /// Write an already encoded WAV buffer to disk.
    ///
    /// # Errors
    ///
    /// Returns an error when the destination cannot be written.
    pub fn write_wav_bytes(&self, output: &Path, wav: &[u8]) -> eyre::Result<()> {
        std::fs::write(output, wav)
            .wrap_err_with(|| format!("failed to write WAV output {}", output.display()))?;
        Ok(())
    }

    /// Return the model sample rate.
    #[must_use]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Keep the existing CLI diagnostic shape while the backend surface narrows.
    #[must_use]
    pub const fn backend_kind(&self) -> &'static str {
        "libtorch"
    }
}

fn encode_pcm16_wav(sample_rate_hz: u32, samples: &[f32]) -> eyre::Result<Vec<u8>> {
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| eyre::eyre!("WAV data size overflows usize"))?;
    let riff_size = 36usize
        .checked_add(data_bytes)
        .ok_or_else(|| eyre::eyre!("WAV RIFF size overflows usize"))?;
    let riff_size =
        u32::try_from(riff_size).map_err(|error| eyre::eyre!("WAV is too large: {error}"))?;
    let data_bytes =
        u32::try_from(data_bytes).map_err(|error| eyre::eyre!("WAV is too large: {error}"))?;
    let byte_rate = sample_rate_hz
        .checked_mul(2)
        .ok_or_else(|| eyre::eyre!("WAV byte rate overflows u32"))?;

    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate_hz.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        let sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The sample was clamped to the signed 16-bit audio range before conversion."
        )]
        let integer = (sample * 32_767.0).round() as i16;
        bytes.extend_from_slice(&integer.to_le_bytes());
    }

    Ok(bytes)
}
