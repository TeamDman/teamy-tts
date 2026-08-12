//! Development-only smoke test for a converted full acoustic checkpoint.

use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use eyre::Result;
use std::path::PathBuf;
use teamy_tts::native_glados::GladosAcousticModel;
use teamy_tts::native_glados::GladosCpuBackend;
use teamy_tts::native_glados::load_acoustic_model_burnpack;
use teamy_tts::native_glados::load_acoustic_model_pytorch;

#[expect(
    clippy::too_many_lines,
    reason = "This diagnostic example prints the complete parity trace in one invocation."
)]
fn main() -> Result<()> {
    let checkpoint = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: verify_glados_acoustic <converted-state-dict.pt>"))?;
    let device = <GladosCpuBackend as Backend>::Device::default();
    let mut model = GladosAcousticModel::<GladosCpuBackend>::init(&device);
    if checkpoint.extension().and_then(|value| value.to_str()) == Some("bpk") {
        load_acoustic_model_burnpack(&mut model, &checkpoint)?;
    } else {
        load_acoustic_model_pytorch(&mut model, &checkpoint)?;
    }

    let tokens =
        Tensor::<GladosCpuBackend, 2, Int>::from_data(TensorData::from([[1_i32, 2, 3]]), &device);
    let speaker = Tensor::<GladosCpuBackend, 2>::zeros([1, 256], &device);
    let output = model.generate(tokens.clone(), speaker.clone(), 1.0);
    let trace = model.trace_mel(
        tokens,
        speaker,
        &output.durations,
        output.pitch.clone(),
        output.energy.clone(),
    );
    let conditioning_values = trace
        .conditioning
        .to_data()
        .to_vec::<f32>()
        .expect("conditioning should contain f32 values");
    let prenet_values = trace
        .prenet
        .to_data()
        .to_vec::<f32>()
        .expect("prenet should contain f32 values");
    let base_conditioning_values = trace
        .base_conditioning
        .to_data()
        .to_vec::<f32>()
        .expect("base conditioning should contain f32 values");
    let pitch_projection_values = trace
        .pitch_projection
        .to_data()
        .to_vec::<f32>()
        .expect("pitch projection should contain f32 values");
    let energy_projection_values = trace
        .energy_projection
        .to_data()
        .to_vec::<f32>()
        .expect("energy projection should contain f32 values");
    let regulated_values = trace
        .regulated
        .to_data()
        .to_vec::<f32>()
        .expect("regulated should contain f32 values");
    let lstm_values = trace
        .lstm_output
        .to_data()
        .to_vec::<f32>()
        .expect("lstm output should contain f32 values");
    let mel_post_values = output
        .mel_post
        .to_data()
        .to_vec::<f32>()
        .expect("mel_post should contain f32 values");
    println!(
        "loaded {} and generated mel {:?}, mel_post {:?}, prenet_head {:?}, prenet_sum {}, base_head {:?}, base_sum {}, pitch_proj_head {:?}, pitch_proj_sum {}, energy_proj_head {:?}, energy_proj_sum {}, conditioning_head {:?}, conditioning_sum {}, regulated_head {:?}, regulated_sum {}, lstm_head {:?}, lstm_sum {}, durations {:?}, pitch_conditions {:?}, pitch {:?}, energy {:?}, mel_head {:?}, mel_post_head {:?}, mel_post_sum {}",
        checkpoint.display(),
        output.mel.dims(),
        output.mel_post.dims(),
        prenet_values.iter().take(5).collect::<Vec<_>>(),
        prenet_values.iter().sum::<f32>(),
        base_conditioning_values.iter().take(5).collect::<Vec<_>>(),
        base_conditioning_values.iter().sum::<f32>(),
        pitch_projection_values.iter().take(5).collect::<Vec<_>>(),
        pitch_projection_values.iter().sum::<f32>(),
        energy_projection_values.iter().take(5).collect::<Vec<_>>(),
        energy_projection_values.iter().sum::<f32>(),
        conditioning_values.iter().take(5).collect::<Vec<_>>(),
        conditioning_values.iter().sum::<f32>(),
        regulated_values.iter().take(5).collect::<Vec<_>>(),
        regulated_values.iter().sum::<f32>(),
        lstm_values.iter().take(5).collect::<Vec<_>>(),
        lstm_values.iter().sum::<f32>(),
        output
            .durations
            .to_data()
            .to_vec::<f32>()
            .expect("durations should contain f32 values"),
        output
            .pitch_conditions
            .to_data()
            .to_vec::<i64>()
            .expect("pitch conditions should contain i64 values"),
        output
            .pitch
            .to_data()
            .to_vec::<f32>()
            .expect("pitch should contain f32 values"),
        output
            .energy
            .to_data()
            .to_vec::<f32>()
            .expect("energy should contain f32 values"),
        output
            .mel
            .to_data()
            .to_vec::<f32>()
            .expect("mel should contain f32 values")
            .into_iter()
            .take(5)
            .collect::<Vec<_>>(),
        mel_post_values.iter().take(5).collect::<Vec<_>>(),
        mel_post_values.iter().sum::<f32>()
    );
    Ok(())
}
