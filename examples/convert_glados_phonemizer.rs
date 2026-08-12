//! Convert the development-time phonemizer checkpoint into a Burnpack.

use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use std::path::PathBuf;
use teamy_tts::frontend_model::GladosPhonemizer;
use teamy_tts::frontend_model::load_glados_phonemizer_pytorch;

fn main() -> eyre::Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let input = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: convert_glados_phonemizer <input.pt> <output.bpk>"))?;
    let output = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| eyre::eyre!("usage: convert_glados_phonemizer <input.pt> <output.bpk>"))?;
    if arguments.next().is_some() {
        return Err(eyre::eyre!(
            "usage: convert_glados_phonemizer <input.pt> <output.bpk>"
        ));
    }

    let device = <burn::backend::NdArray as Backend>::Device::default();
    let mut model = GladosPhonemizer::<burn::backend::NdArray>::init(&device);
    load_glados_phonemizer_pytorch(&mut model, &input)?;
    let mut store = BurnpackStore::from_file(&output)
        .overwrite(true)
        .metadata("format", "teamy-tts-glados-phonemizer")
        .metadata("source", input.display().to_string());
    model.save_into(&mut store)?;
    println!("wrote {}", output.display());
    Ok(())
}
