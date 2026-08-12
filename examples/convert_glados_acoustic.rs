//! Development-only conversion of the upstream acoustic state dictionary.

use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use eyre::Result;
use eyre::eyre;
use std::path::PathBuf;
use teamy_tts::native_glados::GladosAcousticModel;
use teamy_tts::native_glados::GladosCpuBackend;
use teamy_tts::native_glados::load_acoustic_model_pytorch;

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let input = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre!("usage: convert_glados_acoustic <state-dict.pt> <model.bpk>"))?;
    let output = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre!("usage: convert_glados_acoustic <state-dict.pt> <model.bpk>"))?;
    if arguments.next().is_some() {
        return Err(eyre!(
            "usage: convert_glados_acoustic <state-dict.pt> <model.bpk>"
        ));
    }

    let device = <GladosCpuBackend as Backend>::Device::default();
    let mut model = GladosAcousticModel::<GladosCpuBackend>::init(&device);
    load_acoustic_model_pytorch(&mut model, &input)?;
    let mut store = BurnpackStore::from_file(&output)
        .overwrite(true)
        .metadata("format", "teamy-tts-glados-acoustic")
        .metadata("source", input.display().to_string());
    model.save_into(&mut store)?;
    println!("wrote {}", output.display());
    Ok(())
}
