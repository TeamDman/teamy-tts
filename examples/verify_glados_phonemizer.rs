//! Smoke-test a native `DeepPhonemizer` artifact on an unknown word.

use burn::tensor::backend::Backend;
use std::path::PathBuf;
use teamy_tts::frontend_model::GladosPhonemizer;
use teamy_tts::frontend_model::load_glados_phonemizer_burnpack;
use teamy_tts::frontend_model::load_glados_phonemizer_pytorch;

fn main() -> eyre::Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let input = arguments.next().map(PathBuf::from).ok_or_else(|| {
        eyre::eyre!("usage: verify_glados_phonemizer <model.pt|model.bpk> [word]")
    })?;
    let word = arguments
        .next()
        .unwrap_or_else(|| "supercalifragilistic".into())
        .to_string_lossy()
        .into_owned();
    let device = <burn::backend::NdArray as Backend>::Device::default();
    let mut model = GladosPhonemizer::<burn::backend::NdArray>::init(&device);
    if input.extension().and_then(|value| value.to_str()) == Some("bpk") {
        load_glados_phonemizer_burnpack(&mut model, &input)?;
    } else {
        load_glados_phonemizer_pytorch(&mut model, &input)?;
    }
    let trace = model.trace_word(&word)?;
    println!("trace embedding={:?}", &trace.embedding[..5]);
    println!("trace positional={:?}", &trace.positional[..5]);
    println!("trace layer0_query={:?}", &trace.layer0_query[..5]);
    println!("trace layer0_key={:?}", &trace.layer0_key[..5]);
    println!("trace layer0_value={:?}", &trace.layer0_value[..5]);
    println!("trace layer0_weights={:?}", &trace.layer0_weights[..5]);
    println!("trace layer0_context={:?}", &trace.layer0_context[..5]);
    println!(
        "trace layer0_merged_context={:?}",
        &trace.layer0_merged_context[..5]
    );
    println!("trace layer0_attention={:?}", &trace.layer0_attention[..5]);
    println!("trace layer0_norm1={:?}", &trace.layer0_norm1[..5]);
    println!(
        "trace layer0_feed_forward={:?}",
        &trace.layer0_feed_forward[..5]
    );
    for (index, layer) in trace.layers.iter().enumerate() {
        println!("trace layer{index}={:?}", &layer[..5]);
    }
    println!("trace norm={:?}", &trace.norm[..5]);
    println!("trace logits={:?}", &trace.logits[..5]);
    println!(
        "{word}\tids={:?}\t{}",
        model.predict_token_ids(&word)?,
        model.phonemize_word(&word)?
    );
    Ok(())
}
