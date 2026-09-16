use crate::{
    acoustic::ForwardTacotron, cuda::Device, rnn::Dnn, vocoder::HiFiGan, weights::Weights,
};
use anyhow::{Result, ensure};
use std::{path::Path, time::Instant};
fn compare(a: &[f32], b: &[f32]) -> Result<(f64, f32)> {
    ensure!(
        a.len() == b.len() && a.iter().all(|x| x.is_finite()),
        "output shape/finite mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let error = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let power = b.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    Ok((
        (error / power.max(1e-30)).sqrt(),
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max),
    ))
}
pub fn run(args: Vec<String>) -> Result<()> {
    ensure!(
        args.len() == 5,
        "acoustic WEIGHTS FIXTURE VOICE ALPHA ITERATIONS"
    );
    let alpha: f32 = args[3].parse()?;
    let count: usize = args[4].parse()?;
    ensure!(count > 0, "measurement count must be positive");
    let started = Instant::now();
    let weights = Weights::open(Path::new(&args[0]))?;
    let device = Device::new()?;
    let api = Dnn::load(device.clone())?;
    let acoustic = ForwardTacotron::load(&weights, api)?;
    let vocoder = HiFiGan::load(&weights, device.clone())?;
    device.sync()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let fixture = Weights::open(Path::new(&args[1]))?;
    let tokens = fixture.integers("tokens")?;
    let first = Instant::now();
    let output = acoustic.generate(&tokens, &args[2], alpha)?;
    let audio = vocoder
        .synthesize_device(&output.mel, output.frames)?
        .download()?;
    let first_ms = first.elapsed().as_secs_f64() * 1000.;
    let mel_error = compare(&output.mel.download()?, &fixture.values("mel")?)?;
    let raw_mel_error = compare(
        &output.raw_mel.download()?,
        &fixture.values("acoustic.mel")?,
    )?;
    let duration_error = compare(&output.durations, &fixture.values("acoustic.dur")?)?;
    let audio_error = compare(&audio, &fixture.values("audio")?)?;
    let passed = mel_error.0 <= 1e-4 && audio_error.0 <= 1e-3 && audio_error.1 <= 1e-3;
    let mut timings = Vec::new();
    for i in 0..count + 3 {
        let now = Instant::now();
        let out = acoustic.generate(&tokens, &args[2], alpha)?;
        std::hint::black_box(
            vocoder
                .synthesize_device(&out.mel, out.frames)?
                .download()?,
        );
        if i >= 3 {
            timings.push(now.elapsed().as_secs_f64() * 1000.);
        }
    }
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"backend":"native-cuda-f32","stage":"tokens-to-audio","frames":output.frames,"sample_count":audio.len(),"load_ms":load_ms,"first_ms":first_ms,"median_ms":sorted[count/2],"p95_ms":sorted[(count*95).div_ceil(100)-1],"measurement_ms":timings,"mel_relative_rms":mel_error.0,"mel_max_abs":mel_error.1,"raw_mel_relative_rms":raw_mel_error.0,"duration_max_abs":duration_error.1,"audio_relative_rms":audio_error.0,"audio_max_abs":audio_error.1,"correctness_passed":passed,"boundary":"validated host token IDs to complete host PCM; excludes text frontend"})
        )?
    );
    ensure!(passed, "full acoustic/vocoder parity failed");
    Ok(())
}
