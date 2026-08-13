//! Direct Rust bindings to the upstream `GLaDOS` `TorchScript` graphs.
//!
//! This is the migration path away from the handwritten C++ bridge.  It keeps
//! the upstream graph and `LibTorch`'s optimized operators intact while making
//! the process boundary entirely Rust-side through `tch`.

use eyre::Context;
use eyre::bail;
use std::path::Path;
use tch::CModule;
use tch::Device;
use tch::IValue;
use tch::Kind;
use tch::Tensor;

/// A resident upstream acoustic model and GPU vocoder loaded through `tch`.
#[derive(Debug)]
pub struct TchTorchScriptRuntime {
    glados: CModule,
    vocoder: CModule,
    vocoder_device: Device,
    voice_p1: Tensor,
    voice_p2: Tensor,
}

impl TchTorchScriptRuntime {
    /// Load the upstream `TorchScript` pair and prepared voice artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error when the model files, voice embeddings, or configured
    /// CUDA device cannot be loaded.
    pub fn from_model_dir(
        model_dir: &Path,
        voice_p1_path: &Path,
        voice_p2_path: &Path,
    ) -> eyre::Result<Self> {
        let glados_path = model_dir.join("glados-new.pt");
        let vocoder_path = model_dir.join("vocoder-gpu.pt");
        if !glados_path.is_file() {
            bail!(
                "TorchScript GLaDOS model is missing: {}",
                glados_path.display()
            );
        }
        if !vocoder_path.is_file() {
            bail!(
                "TorchScript vocoder model is missing: {}",
                vocoder_path.display()
            );
        }

        let device_index = crate::config::effective_torch_device()?.unwrap_or(0);
        let vocoder_device = if device_index < 0 {
            tracing::warn!(device_index, "using CPU for tch TorchScript vocoder");
            Device::Cpu
        } else {
            let device_index = usize::try_from(device_index)
                .wrap_err("TEAMY_TTS_TORCH_DEVICE must be a non-negative CUDA device index")?;
            let cuda_available = tch::Cuda::is_available();
            let cuda_device_count = tch::Cuda::device_count();
            tracing::info!(
                cuda_available,
                cuda_device_count,
                device_index,
                "checking tch CUDA runtime"
            );
            if !cuda_available {
                bail!(
                    "CUDA is unavailable to tch/LibTorch; use --torch-device -1 or set TEAMY_TTS_TORCH_DEVICE=-1 for CPU"
                );
            }
            if i64::try_from(device_index).is_ok_and(|index| index >= cuda_device_count) {
                bail!(
                    "requested CUDA device {device_index}, but LibTorch reports {cuda_device_count} device(s)"
                );
            }
            Device::Cuda(device_index)
        };

        tracing::info!(path = %glados_path.display(), "loading acoustic TorchScript module through tch");
        let mut glados = CModule::load_on_device(&glados_path, Device::Cpu)
            .wrap_err_with(|| format!("failed to load {}", glados_path.display()))?;
        tracing::info!(path = %vocoder_path.display(), device = ?vocoder_device, "loading vocoder TorchScript module through tch");
        let mut vocoder =
            CModule::load_on_device(&vocoder_path, vocoder_device).wrap_err_with(|| {
                format!(
                    "failed to load {} on {vocoder_device:?}",
                    vocoder_path.display()
                )
            })?;
        glados.set_eval();
        vocoder.set_eval();

        let runtime = Self {
            glados,
            vocoder,
            vocoder_device,
            voice_p1: load_voice_embedding(voice_p1_path)?,
            voice_p2: load_voice_embedding(voice_p2_path)?,
        };
        runtime.warm_up()?;
        Ok(runtime)
    }

    fn warm_up(&self) -> eyre::Result<()> {
        let warmup_tokens = [97_i32, 24, 106, 27, 20, 5, 10, 79, 72, 28, 68, 54, 6];
        tracing::debug!("warming up tch TorchScript models");
        for _ in 0..2 {
            let _ = self.synthesize(&warmup_tokens, "p2", 1.0)?;
        }
        tracing::debug!("tch TorchScript model warmup complete");
        Ok(())
    }

    /// Generate audio through the loaded upstream graph.
    ///
    /// # Errors
    ///
    /// Returns an error when the voice is unknown, the `TorchScript` graph fails,
    /// or the generated waveform cannot be copied to the host.
    pub fn synthesize(
        &self,
        token_values: &[i32],
        voice: &str,
        alpha: f32,
    ) -> eyre::Result<Vec<f32>> {
        let speaker = match voice {
            "p1" => &self.voice_p1,
            "p2" => &self.voice_p2,
            other => bail!("unknown voice {other:?}; expected p1 or p2"),
        };
        let token_values = token_values
            .iter()
            .map(|&token| i64::from(token))
            .collect::<Vec<_>>();
        let tokens = Tensor::from_slice(&token_values).reshape([1, -1]);
        let output = self
            .glados
            .method_is(
                "generate_jit",
                &[
                    IValue::Tensor(tokens),
                    IValue::Tensor(speaker.shallow_clone()),
                    IValue::Double(f64::from(alpha)),
                ],
            )
            .wrap_err("GLaDOS generate_jit failed through tch")?;
        let mel = mel_post(output)?;
        let audio = self
            .vocoder
            .forward_ts(&[mel.to_device(self.vocoder_device)])
            .wrap_err("HiFiGAN forward failed through tch")?
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous()
            .view([-1]);
        Vec::<f32>::try_from(audio).wrap_err("failed to copy tch waveform to the host")
    }
}

fn mel_post(output: IValue) -> eyre::Result<Tensor> {
    let IValue::GenericDict(entries) = output else {
        bail!("GLaDOS output was not a TorchScript generic dictionary");
    };
    for (key, value) in entries {
        if matches!(key, IValue::String(ref key) if key == "mel_post") {
            return Tensor::try_from(value).wrap_err("GLaDOS mel_post was not a tensor");
        }
    }
    bail!("GLaDOS output did not contain mel_post")
}

fn load_voice_embedding(path: &Path) -> eyre::Result<Tensor> {
    let bytes = std::fs::read(path)
        .wrap_err_with(|| format!("failed to read voice embedding {}", path.display()))?;
    let expected_bytes =
        crate::native_glados::GLADOS_SPEAKER_EMBEDDING_DIMENSION * size_of::<f32>();
    if bytes.len() != expected_bytes {
        bail!(
            "voice embedding {} has {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected_bytes
        );
    }
    let values = bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let bytes = <[u8; 4]>::try_from(chunk).expect("chunks_exact guarantees four bytes");
            f32::from_le_bytes(bytes)
        })
        .collect::<Vec<_>>();
    let embedding_width =
        i64::try_from(values.len()).wrap_err("voice embedding width exceeds i64")?;
    Ok(Tensor::from_slice(&values).reshape([1, embedding_width]))
}
