use crate::{cuda::Device, phonemizer::Phonemizer, weights::Weights};
use anyhow::{Result, ensure};
use std::{path::Path, time::Instant};
pub fn run(args: Vec<String>) -> Result<()> {
    ensure!(
        args.len() == 4,
        "phonemizer WEIGHTS FIXTURE WORD ITERATIONS"
    );
    let count: usize = args[3].parse()?;
    ensure!(count > 0, "positive measurement count required");
    let started = Instant::now();
    let weights = Weights::open(Path::new(&args[0]))?;
    let device = Device::new()?;
    let model = Phonemizer::load(&weights, device.clone())?;
    device.sync()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let fixture = Weights::open(Path::new(&args[1]))?;
    let tokens = Phonemizer::tokens(&args[2])?;
    ensure!(
        tokens == fixture.integers("tokens")?,
        "word tokens differ from oracle"
    );
    let start = Instant::now();
    let output = model.forward(&tokens)?;
    let logits = output.transpose(53, tokens.len())?.download()?;
    let first_ms = start.elapsed().as_secs_f64() * 1000.;
    let expected = fixture.values("logits")?;
    ensure!(
        logits.len() == expected.len() && logits.iter().all(|v| v.is_finite()),
        "invalid logits"
    );
    let power = expected.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    let error = logits
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let max_abs = logits
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let relative_rms = (error / power.max(1e-30)).sqrt();
    let indices = output.argmax(tokens.len(), 53)?.download_i32()?;
    let exact_indices = indices == fixture.integers("indices")?;
    let phonemes = Phonemizer::decode(&indices)?;
    let passed = exact_indices && relative_rms <= 1e-4 && max_abs <= 1e-3;
    let mut times = Vec::new();
    for i in 0..count + 3 {
        let start = Instant::now();
        std::hint::black_box(model.phonemize_word(&args[2])?);
        if i >= 3 {
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"backend":"native-cuda-f32","stage":"phonemizer","word":args[2],"phonemes":phonemes,"load_ms":load_ms,"first_logits_ms":first_ms,"median_ms":sorted[count/2],"p95_ms":sorted[(count*95).div_ceil(100)-1],"measurement_ms":times,"relative_rms":relative_rms,"max_abs":max_abs,"exact_indices":exact_indices,"correctness_passed":passed})
        )?
    );
    ensure!(passed, "phonemizer parity failed");
    Ok(())
}
