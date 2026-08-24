//! Line-oriented stdin synthesis with synchronous local playback.

use crate::audio;
use crate::cli::output::CliOutput;
use crate::cli::say::emit_output_path;
use crate::cli::say::load_runtime;
use crate::cli::say::resolve_output_path;
use crate::cli::say::synthesize_to_wav;
use crate::cli::say::write_wav_output;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::io::BufRead;
use std::io::IsTerminal;
use std::io::Write;
use std::io::{self};
use std::thread::JoinHandle;
use std::time::Instant;
use teamy_cancellation::CancellationToken;
use tokio::sync::mpsc;
use tokio::time::Duration;

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

type StdinMessage = Result<String, String>;

/// Read stdin lines, play each result synchronously, and only persist output
/// when `--output-dir` is explicitly provided.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct InteractiveArgs {
    /// Stable model identifier. Defaults to glados.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,

    /// Speaker embedding to use. Defaults to p2.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub voice: Option<String>,

    /// Duration/pitch scaling factor. Defaults to 1.0.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub alpha: Option<f32>,

    /// Playback/output amplitude multiplier in the inclusive range 0.0..=1.0.
    /// Defaults to 1.0.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub volume: Option<f32>,

    /// Compatibility selector for the only inference backend: tch/LibTorch.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub backend: Option<String>,

    /// Interpret each input line as `GLaDOS` IPA-like phoneme symbols.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub phonemes: bool,

    /// Directory for automatically numbered WAV outputs. Supplying this flag
    /// opts into persistent output files.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub output_dir: Option<String>,
}

impl InteractiveArgs {
    /// # Errors
    ///
    /// Returns an error if the model is unavailable, a line cannot be
    /// synthesized or written, or the operating system cannot play it.
    pub async fn invoke(self, cancellation_token: CancellationToken) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let voice = self.voice.unwrap_or_else(|| "p2".to_string());
        let alpha = self.alpha.unwrap_or(1.0);
        let volume = self.volume.unwrap_or(1.0);
        let output_dir = self.output_dir.as_deref();
        let (_model, runtime) = load_runtime(model_id, self.backend.as_deref())?;
        cancellation_token.bail_if_cancelled()?;
        let stdin_is_terminal = io::stdin().is_terminal();
        let (mut input_messages, reader_handle) = spawn_stdin_reader()?;

        tracing::info!(
            voice = %voice,
            "interactive mode ready; enter text, or send EOF to exit"
        );
        loop {
            cancellation_token.bail_if_cancelled()?;
            if stdin_is_terminal {
                eprint!("> ");
                io::stderr().flush()?;
            }

            let Some(message) = (tokio::select! {
                biased;
                () = wait_for_cancellation(cancellation_token.clone()) => {
                    tracing::info!("interactive mode cancelled while waiting for stdin");
                    drop(input_messages);
                    drop(reader_handle);
                    cancellation_token.bail_if_cancelled()?;
                    unreachable!("the cancellation watcher completed without cancellation");
                }
                message = input_messages.recv() => message,
            }) else {
                join_stdin_reader(reader_handle)?;
                break;
            };

            // Cancellation and stdin delivery can become ready in the same
            // scheduler turn. Re-check after receiving a line so Ctrl+C never
            // admits a line into synthesis after cancellation was requested.
            cancellation_token.bail_if_cancelled()?;
            let line = message.map_err(|error| eyre::eyre!(error))?;
            let text = line.trim_end_matches(['\r', '\n']);
            if text.trim().is_empty() {
                continue;
            }

            let output = resolve_output_path(text, output_dir, None)?;
            let wav = synthesize_to_wav(&runtime, text, self.phonemes, &voice, alpha, volume)?;
            if let Some(output) = output.as_deref() {
                write_wav_output(&runtime, output, &wav)?;
                emit_output_path(output)?;
                tracing::info!(output = %output.display(), "playing interactive WAV output");
            } else {
                tracing::info!("playing interactive in-memory WAV output");
            }
            let playback_started = Instant::now();
            audio::play_wav_bytes(&wav)?;
            tracing::info!(
                elapsed_ms = playback_started.elapsed().as_millis(),
                "interactive playback complete"
            );
        }

        tracing::info!("interactive mode finished");
        Ok(CliOutput::none())
    }
}

fn spawn_stdin_reader() -> Result<(mpsc::UnboundedReceiver<StdinMessage>, JoinHandle<()>)> {
    let (sender, receiver) = mpsc::unbounded_channel();
    let handle = std::thread::Builder::new()
        .name(String::from("teamy-tts-stdin-reader"))
        .spawn(move || {
            let stdin = io::stdin();
            let mut input = stdin.lock();
            let mut line = String::new();
            loop {
                line.clear();
                let message = match input.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => Ok(line.clone()),
                    Err(error) => Err(error.to_string()),
                };
                let should_stop = message.is_err() || sender.send(message).is_err();
                if should_stop {
                    break;
                }
            }
        })?;
    Ok((receiver, handle))
}

fn join_stdin_reader(handle: JoinHandle<()>) -> Result<()> {
    handle
        .join()
        .map_err(|panic| eyre::eyre!("stdin reader thread panicked: {panic:?}"))
}

async fn wait_for_cancellation(cancellation_token: CancellationToken) {
    while !cancellation_token.is_cancelled() {
        tokio::time::sleep(CANCELLATION_POLL_INTERVAL).await;
    }
}
