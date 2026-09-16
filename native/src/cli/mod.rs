mod acoustic;
mod phonemizer;
mod rnn;
mod vocoder;
pub fn run() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("vocoder") => vocoder::run(args.collect()),
        Some("rnn") => rnn::run(args.collect()),
        Some("acoustic") => acoustic::run(args.collect()),
        Some("phonemizer") => phonemizer::run(args.collect()),
        _ => anyhow::bail!("usage: teamy-glados-native vocoder WEIGHTS FIXTURE [ITERATIONS]"),
    }
}
