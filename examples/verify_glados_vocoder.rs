//! Development-only smoke test for a converted `GLaDOS` vocoder.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use eyre::Result;
use std::path::PathBuf;
use teamy_tts::native_glados::GladosCpuBackend;
use teamy_tts::native_glados::vocoder::HiFiGan;
use teamy_tts::native_glados::vocoder::load_hifigan_burnpack;
use teamy_tts::native_glados::vocoder::load_hifigan_pytorch;

fn main() -> Result<()> {
    let checkpoint = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: verify_glados_vocoder <model.pt|model.bpk>"))?;
    let device = <GladosCpuBackend as Backend>::Device::default();
    let mut model = HiFiGan::<GladosCpuBackend>::init(&device);
    if checkpoint.extension().and_then(|value| value.to_str()) == Some("bpk") {
        load_hifigan_burnpack(&mut model, &checkpoint)?;
    } else {
        load_hifigan_pytorch(&mut model, &checkpoint)?;
    }
    let mel = Tensor::<GladosCpuBackend, 3>::zeros([1, 80, 1], &device);
    let audio = model.forward(mel);
    let values = audio
        .to_data()
        .to_vec::<f32>()
        .expect("audio should contain f32 values");
    println!(
        "loaded {} and generated audio {:?}, head {:?}, sum {}",
        checkpoint.display(),
        audio.dims(),
        values.iter().take(5).collect::<Vec<_>>(),
        values.iter().sum::<f32>()
    );
    Ok(())
}
