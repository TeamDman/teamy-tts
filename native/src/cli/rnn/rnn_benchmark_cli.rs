use crate::{
    cuda::Device,
    rnn::{Dnn, Rnn},
    weights::Weights,
};
use anyhow::{Result, ensure};
use std::{path::Path, time::Instant};
pub fn run(args: Vec<String>) -> Result<()> {
    ensure!(args.len() == 3, "rnn WEIGHTS FIXTURE PREFIX");
    let weights = Weights::open(Path::new(&args[0]))?;
    let fixture = Weights::open(Path::new(&args[1]))?;
    let shape = fixture.shape("input")?;
    ensure!(shape.len() == 3 && shape[0] == 1, "RNN fixture shape");
    let time = shape[1];
    let device = Device::new()?;
    let api = Dnn::load(device.clone())?;
    let load = Instant::now();
    let model = Rnn::load(
        &weights,
        api,
        &format!("acoustic.{}", args[2]),
        args[2] == "lstm",
    )?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.;
    let x = device.upload_f32(&fixture.values("input")?)?;
    let reference = fixture.values("output")?;
    let first = Instant::now();
    let output = model.run(&x, time)?.download()?;
    let first_ms = first.elapsed().as_secs_f64() * 1000.;
    ensure!(
        output.len() == reference.len() && output.iter().all(|v| v.is_finite()),
        "RNN output shape/finite"
    );
    let max_abs = output
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let error = output
        .iter()
        .zip(&reference)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let power = reference.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
    let relative_rms = (error / power.max(1e-30)).sqrt();
    let mut times = Vec::new();
    for _ in 0..3 {
        model.run(&x, time)?.download()?;
    }
    for _ in 0..20 {
        let start = Instant::now();
        model.run(&x, time)?.download()?;
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let passed = relative_rms <= 1e-4 && max_abs <= 1e-4;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"stage":args[2],"load_ms":load_ms,"first_ms":first_ms,"median_ms":sorted[10],"measurement_ms":times,"max_abs":max_abs,"relative_rms":relative_rms,"correctness_passed":passed,"boundary":"resident device input to host RNN output; excludes text/model loading"})
        )?
    );
    ensure!(passed, "RNN parity failed");
    Ok(())
}
