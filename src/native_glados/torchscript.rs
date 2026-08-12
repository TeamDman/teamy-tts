//! Optional CUDA `TorchScript` runtime for the upstream `GLaDOS` checkpoints.
//!
//! The default runtime is pure Rust/Burn. This module is deliberately behind
//! the `torchscript` feature because it requires a matching LibTorch/PyTorch
//! installation at build and run time. The C++ bridge keeps the upstream
//! `TorchScript` graph intact, which is currently the practical fast path on a
//! CUDA-equipped development machine.

use eyre::Context;
use eyre::bail;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

unsafe extern "C" {
    fn teamy_tts_torchscript_create(
        glados_path: *const c_char,
        vocoder_path: *const c_char,
        device_index: i32,
        error: *mut *mut c_char,
    ) -> *mut c_void;
    fn teamy_tts_torchscript_synthesize(
        runtime: *mut c_void,
        token_values: *const i64,
        token_count: usize,
        speaker_values: *const f32,
        speaker_count: usize,
        alpha: f32,
        audio_values: *mut *mut f32,
        audio_count: *mut usize,
        error: *mut *mut c_char,
    ) -> i32;
    fn teamy_tts_torchscript_free_audio(audio_values: *mut f32);
    fn teamy_tts_torchscript_destroy(runtime: *mut c_void);
    fn teamy_tts_torchscript_free_error(error: *mut c_char);
}

/// A loaded upstream `GLaDOS` acoustic model and `HiFiGAN` vocoder.
#[derive(Debug)]
pub struct TorchScriptRuntime {
    raw: NonNull<c_void>,
    voice_p1: Vec<f32>,
    voice_p2: Vec<f32>,
}

impl TorchScriptRuntime {
    /// Load the upstream `TorchScript` pair and the prepared voice artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error when model paths are not representable as C strings,
    /// voice artifacts are malformed, or `LibTorch` rejects either module.
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

        let glados_path = path_cstring(&glados_path)?;
        let vocoder_path = path_cstring(&vocoder_path)?;
        let voice_p1 = read_voice_embedding(voice_p1_path)?;
        let voice_p2 = read_voice_embedding(voice_p2_path)?;
        let device_index = crate::config::effective_torch_device()?.unwrap_or(0);

        let mut error = std::ptr::null_mut();
        // SAFETY: The strings remain alive for the duration of this call and
        // the bridge returns an owned runtime or a separately owned error.
        let raw = unsafe {
            teamy_tts_torchscript_create(
                glados_path.as_ptr(),
                vocoder_path.as_ptr(),
                device_index,
                &raw mut error,
            )
        };
        let Some(raw) = NonNull::new(raw) else {
            return Err(eyre::eyre!(take_error(&mut error)));
        };

        let runtime = Self {
            raw,
            voice_p1,
            voice_p2,
        };
        runtime.warm_up()?;
        Ok(runtime)
    }

    fn warm_up(&self) -> eyre::Result<()> {
        let warmup_tokens = [97_i32, 24, 106, 27, 20, 5, 10, 79, 72, 28, 68, 54, 6];
        tracing::debug!("warming up TorchScript CUDA models");
        for _ in 0..2 {
            let _ = self.synthesize(&warmup_tokens, "p2", 1.0)?;
        }
        tracing::debug!("TorchScript CUDA model warmup complete");
        Ok(())
    }

    /// Generate audio through the loaded upstream graph.
    ///
    /// # Errors
    ///
    /// Returns an error when the voice is unknown or the bridge reports an
    /// inference failure.
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
        let mut audio = std::ptr::null_mut();
        let mut audio_count = 0;
        let mut error = std::ptr::null_mut();
        // SAFETY: All input slices and the runtime remain valid for this call.
        // The bridge allocates the returned audio buffer, which is released by
        // the matching free function below.
        let status = unsafe {
            teamy_tts_torchscript_synthesize(
                self.raw.as_ptr(),
                token_values.as_ptr(),
                token_values.len(),
                speaker.as_ptr(),
                speaker.len(),
                alpha,
                &raw mut audio,
                &raw mut audio_count,
                &raw mut error,
            )
        };
        if status != 0 {
            return Err(eyre::eyre!(take_error(&mut error)));
        }
        let Some(audio) = NonNull::new(audio) else {
            bail!("TorchScript bridge returned no audio samples");
        };
        // SAFETY: The bridge returned `audio_count` contiguous f32 values and
        // ownership remains with the bridge until the matching free call.
        let values = unsafe { std::slice::from_raw_parts(audio.as_ptr(), audio_count).to_vec() };
        // SAFETY: `audio` was allocated by the bridge for this exact purpose.
        unsafe { teamy_tts_torchscript_free_audio(audio.as_ptr()) };
        Ok(values)
    }
}

impl Drop for TorchScriptRuntime {
    fn drop(&mut self) {
        // SAFETY: `raw` is the live runtime returned by the matching create
        // function and this is its single owner.
        unsafe { teamy_tts_torchscript_destroy(self.raw.as_ptr()) };
    }
}

fn path_cstring(path: &Path) -> eyre::Result<CString> {
    let path = path
        .to_str()
        .ok_or_else(|| eyre::eyre!("path is not valid UTF-8: {}", path.display()))?;
    CString::new(path).wrap_err_with(|| format!("path contains an interior NUL: {path:?}"))
}

fn read_voice_embedding(path: &Path) -> eyre::Result<Vec<f32>> {
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
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let bytes =
                <[u8; 4]>::try_from(chunk).expect("chunks_exact guarantees four-byte voice values");
            Ok(f32::from_le_bytes(bytes))
        })
        .collect()
}

fn take_error(error: &mut *mut c_char) -> String {
    let Some(error_ptr) = NonNull::new(*error) else {
        return "unknown TorchScript bridge error".to_string();
    };
    // SAFETY: The bridge returns a NUL-terminated owned error string.
    let message = unsafe { CStr::from_ptr(error_ptr.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: The pointer was allocated by the bridge's matching allocator.
    unsafe { teamy_tts_torchscript_free_error(error_ptr.as_ptr()) };
    *error = std::ptr::null_mut();
    message
}
