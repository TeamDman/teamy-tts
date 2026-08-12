//! Convert one development-time predictor checkpoint into a Burnpack.
//!
//! The input checkpoint is produced by `tools/export_glados_predictor.py`.
//! This example is intentionally separate from the packaged CLI so the runtime
//! remains independent of Python and `TorchScript`.

use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use std::env;
use std::path::PathBuf;
use teamy_tts::native_glados::GladosCpuBackend;
use teamy_tts::native_glados::SeriesPredictorConfig;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: convert_glados_predictor <input.pt> <output.bpk>"))?;
    let output = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: convert_glados_predictor <input.pt> <output.bpk>"))?;
    let predictor_name = arguments
        .next()
        .unwrap_or_else(|| "pitch_cond_pred".into())
        .to_string_lossy()
        .into_owned();

    let device: <GladosCpuBackend as Backend>::Device =
        <GladosCpuBackend as Backend>::Device::default();
    let (conditional, hidden_dimension, output_dimension) = match predictor_name.as_str() {
        "dur_pred" => (true, 128, 1),
        "pitch_cond_pred" => (false, 128, 3),
        "pitch_pred" => (true, 256, 1),
        "energy_pred" => (false, 64, 1),
        other => return Err(eyre::eyre!("unknown predictor {other:?}")),
    };
    let config =
        SeriesPredictorConfig::new(135, 128, 256, 3, 256, 5, hidden_dimension, output_dimension);
    let mut store = BurnpackStore::from_file(&output)
        .overwrite(true)
        .metadata("format", "teamy-tts-glados-predictor")
        .metadata("predictor", predictor_name.clone())
        .metadata("source", input.display().to_string());
    if conditional {
        let mut predictor = config.init_conditional::<GladosCpuBackend>(4, 4, &device);
        teamy_tts::native_glados::load_conditional_series_predictor_pytorch(
            &mut predictor,
            &input,
        )?;
        predictor.save_into(&mut store)?;
    } else {
        let mut predictor = config.init_series::<GladosCpuBackend>(&device);
        teamy_tts::native_glados::load_series_predictor_pytorch(&mut predictor, &input)?;
        predictor.save_into(&mut store)?;
    }
    println!("wrote {}", output.display());
    Ok(())
}
