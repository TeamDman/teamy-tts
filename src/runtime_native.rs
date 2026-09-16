//! Source-defined CUDA GLaDOS runtime and the shared product frontend.
use crate::frontend::GladosFrontend;
use eyre::{Context, Result};
use std::path::{Path, PathBuf};
use teamy_glados_native::{Engine, TextModel};

pub const FRONTEND_ROLE: &str = "frontend-dictionary";
pub const VOICE_P1_ROLE: &str = "voice-p1";
pub const VOICE_P2_ROLE: &str = "voice-p2";

pub fn configured_model_dir() -> Result<PathBuf> {
    crate::config::effective_native_model_dir()?.ok_or_else(||eyre::eyre!("set TEAMY_TTS_NATIVE_MODEL_DIR or configure --native-model-dir with the exported native model directory"))
}

#[derive(Debug)]
pub struct GladosTextFrontend {
    frontend: GladosFrontend,
    phonemizer: TextModel,
}
impl GladosTextFrontend {
    pub fn from_native(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            frontend: GladosFrontend::from_tsv(&model_dir.join("frontend.tsv"))?,
            phonemizer: TextModel::from_file(&model_dir.join("weights.safetensors"))
                .map_err(|e| eyre::eyre!("{e:#}"))?,
        })
    }
    pub fn phonemize(&self, text: &str) -> Result<String> {
        self.frontend.phonemize_with(text, |word| {
            self.phonemizer
                .phonemize_word(word)
                .map_err(|e| eyre::eyre!("{e:#}"))
        })
    }
    pub fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
        self.frontend.tokenize_phonemes(&self.phonemize(text)?)
    }
    pub fn tokenize_phonemes(&self, text: &str) -> Result<Vec<i32>> {
        self.frontend.tokenize_phonemes(text)
    }
}

#[derive(Debug)]
pub struct GladosRuntime {
    frontend: GladosTextFrontend,
    engine: Engine,
}
impl GladosRuntime {
    pub fn from_native(model_dir: &Path) -> Result<Self> {
        let frontend = GladosTextFrontend::from_native(model_dir)?;
        let engine = Engine::from_file(&model_dir.join("weights.safetensors"))
            .map_err(|e| eyre::eyre!("{e:#}"))?;
        let runtime = Self { frontend, engine };
        // Pay recurrent and matrix-library initialization before declaring ready,
        // including the unknown-word path used by interactive sessions.
        runtime
            .frontend
            .phonemizer
            .phonemize_word("quux")
            .map_err(|e| eyre::eyre!("{e:#}"))?;
        runtime.synthesize("Hello, friend", "p2", 1.)?;
        Ok(runtime)
    }
    pub fn synthesize(&self, text: &str, voice: &str, alpha: f32) -> Result<Vec<f32>> {
        let tokens = self.frontend.tokenize_text(text)?;
        self.engine
            .synthesize(&tokens, voice, alpha)
            .map_err(|e| eyre::eyre!("{e:#}"))
    }
    pub fn synthesize_phonemes(&self, text: &str, voice: &str, alpha: f32) -> Result<Vec<f32>> {
        let tokens = self.frontend.tokenize_phonemes(text)?;
        self.engine
            .synthesize(&tokens, voice, alpha)
            .map_err(|e| eyre::eyre!("{e:#}"))
    }
    pub fn write_wav(&self, output: &Path, samples: &[f32]) -> Result<()> {
        self.write_wav_bytes(output, &self.wav_bytes(samples)?)
    }
    pub fn wav_bytes(&self, samples: &[f32]) -> Result<Vec<u8>> {
        crate::runtime_wav::encode_pcm16_wav(self.sample_rate_hz(), samples)
    }
    pub fn write_wav_bytes(&self, output: &Path, wav: &[u8]) -> Result<()> {
        std::fs::write(output, wav).wrap_err_with(|| format!("write WAV {}", output.display()))
    }
    pub const fn sample_rate_hz(&self) -> u32 {
        22050
    }
    pub const fn backend_kind(&self) -> &'static str {
        "cuda-native"
    }
}
