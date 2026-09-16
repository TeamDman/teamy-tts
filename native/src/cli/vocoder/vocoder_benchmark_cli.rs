use crate::{cuda::Device, vocoder::HiFiGan, weights::Weights};
use anyhow::{Result, ensure};
use std::{path::Path, time::Instant};

pub fn run(args: Vec<String>) -> Result<()> {
    ensure!(
        (2..=3).contains(&args.len()),
        "vocoder WEIGHTS FIXTURE [ITERATIONS]"
    );
    let iterations: usize = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(10);
    ensure!(iterations > 0, "iterations must be positive");
    let started = Instant::now();
    let weights = Weights::open(Path::new(&args[0]))?;
    let read_ms = started.elapsed().as_secs_f64() * 1000.;
    let device = Device::new()?;
    let device_ms = started.elapsed().as_secs_f64() * 1000. - read_ms;
    let model = HiFiGan::load(&weights, device.clone())?;
    device.sync()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let fixture = Weights::open(Path::new(&args[1]))?;
    let shape = fixture.shape("mel")?;
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] == 80,
        "fixture mel shape"
    );
    let frames = shape[2];
    let mel = fixture.values("mel")?;
    let reference = fixture.values("audio")?;
    let first = Instant::now();
    let output = model.synthesize(&mel, frames)?;
    let first_ms = first.elapsed().as_secs_f64() * 1000.;
    ensure!(output.len() == reference.len(), "sample count mismatch");
    ensure!(output.iter().all(|x| x.is_finite()), "nonfinite output");
    let error = output
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let power = reference.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    let relative_rms = (error / power.max(1e-30)).sqrt();
    let max_abs = output
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let passed = relative_rms <= 1e-3 && max_abs <= 1e-3;
    let mut timings = Vec::new();
    for _ in 0..3 {
        model.synthesize(&mel, frames)?;
    }
    for _ in 0..iterations {
        let start = Instant::now();
        let audio = model.synthesize(&mel, frames)?;
        std::hint::black_box(audio);
        timings.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "backend":"native-cuda-f32", "stage":"vocoder", "frames":frames,
            "sample_count":output.len(),"read_ms":read_ms,"device_ms":device_ms,"load_ms":load_ms,
            "first_ms":first_ms,"measurement_ms":timings,"median_ms":sorted[iterations/2],
            "p95_ms":sorted[(iterations*95).div_ceil(100)-1],
            "relative_rms":relative_rms,"max_abs":max_abs,"correctness_passed":passed,
            "correctness_gate":"exact sample count, finite, relative RMS <= 0.001 and max absolute error <= 0.001",
            "timing_boundary":"host mel to complete host PCM; includes transfers and allocation"
        }))?
    );
    ensure!(passed, "vocoder waveform parity failed");
    Ok(())
}
