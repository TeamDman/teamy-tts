//! LibTorch-owned GLaDOS runtime components.

pub mod tch_torchscript;

pub const GLADOS_SPEAKER_EMBEDDING_DIMENSION: usize = 256;

#[cfg(all(windows, teamy_tts_cuda_link))]
unsafe extern "C" {
    fn teamy_tts_force_torch_cuda();
}

#[cfg(all(windows, teamy_tts_cuda_link))]
#[used]
static FORCE_TORCH_CUDA_LINK: unsafe extern "C" fn() = teamy_tts_force_torch_cuda;
