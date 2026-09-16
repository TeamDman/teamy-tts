//! Source-defined GLaDOS inference. Only numerical tensors are loaded at runtime.
mod acoustic;
mod cli;
mod cuda;
#[cfg(test)]
mod cuda_tests;
mod phonemizer;
mod rnn;
mod vocoder;
mod weights;

use anyhow::Result;
use std::path::Path;

/// Entrypoint for the development-only stage benchmark executable.
pub fn run_benchmarks() -> Result<()> {
    cli::run()
}

pub fn check_device() -> Result<()> {
    let device = cuda::Device::new()?;
    let _api = rnn::Dnn::load(device)?;
    Ok(())
}

pub struct Engine {
    acoustic: acoustic::ForwardTacotron,
    vocoder: vocoder::HiFiGan,
}
impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeGladosEngine").finish_non_exhaustive()
    }
}
impl Engine {
    pub fn from_file(path: &Path) -> Result<Self> {
        let weights = weights::Weights::open(path)?;
        let device = cuda::Device::new()?;
        let api = rnn::Dnn::load(device.clone())?;
        let acoustic = acoustic::ForwardTacotron::load(&weights, api)?;
        let vocoder = vocoder::HiFiGan::load(&weights, device)?;
        Ok(Self { acoustic, vocoder })
    }
    pub fn synthesize(&self, tokens: &[i32], voice: &str, alpha: f32) -> Result<Vec<f32>> {
        let output = self.acoustic.generate(tokens, voice, alpha)?;
        self.vocoder
            .synthesize_device(&output.mel, output.frames)?
            .download()
    }
}

pub struct TextModel(phonemizer::Phonemizer);
impl std::fmt::Debug for TextModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeGladosPhonemizer")
            .finish_non_exhaustive()
    }
}
impl TextModel {
    pub fn from_file(path: &Path) -> Result<Self> {
        let weights = weights::Weights::open(path)?;
        let device = cuda::Device::new()?;
        Ok(Self(phonemizer::Phonemizer::load(&weights, device)?))
    }
    pub fn phonemize_word(&self, word: &str) -> Result<String> {
        self.0.phonemize_word(word)
    }
}
