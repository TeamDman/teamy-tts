use crate::audio;
use crate::cli::model_preparation_hint;
use crate::cli::output::CliOutput;
use crate::model_registry;
use crate::runtime::GladosRuntime;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

const MAX_DEFAULT_OUTPUT_STEM_CHARS: usize = 80;

/// Synthesize one utterance and play it. Audio is only written when an output
/// destination is explicitly provided.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct SayArgs {
    /// Text to synthesize.
    #[facet(args::positional)]
    pub text: String,

    /// Stable model identifier. Defaults to `glados`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub model: Option<String>,

    /// Speaker embedding to use. Defaults to `p2`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub voice: Option<String>,

    /// Duration/pitch scaling factor. Defaults to `1.0`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub alpha: Option<f32>,

    /// Playback/output amplitude multiplier in the inclusive range 0.0..=1.0.
    /// Defaults to `1.0`.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub volume: Option<f32>,

    /// Compatibility selector for the only inference backend: tch/LibTorch.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub backend: Option<String>,

    /// Interpret the positional input as `GLaDOS` IPA-like phoneme symbols.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub phonemes: bool,

    /// Output filename or path. With --output-dir, this is relative to that
    /// directory; without it, the value is used as the complete path.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub output: Option<String>,

    /// Directory for automatic numbered outputs, or for --output filenames.
    /// Supplying this flag opts into persistent output files.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub output_dir: Option<String>,
}

/// Synthesize one utterance into a WAV file without playing it.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct WriteArgs {
    /// Text to synthesize.
    #[facet(args::positional)]
    pub text: String,

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

    /// Compatibility selector for the only inference backend: tch/LibTorch.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub backend: Option<String>,

    /// Interpret the positional input as `GLaDOS` IPA-like phoneme symbols.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub phonemes: bool,

    /// Output filename or path. With --output-dir, this is relative to that
    /// directory; without it, the value is used as the complete path.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub output: Option<String>,

    /// Directory for automatic numbered outputs, or for --output filenames.
    /// Supplying this flag opts into persistent output files.
    #[facet(args::named)]
    #[arbitrary(default)]
    pub output_dir: Option<String>,
}

impl SayArgs {
    /// # Errors
    ///
    /// Returns an error if the selected model is unknown or unprepared, the
    /// text cannot be phonemized, or inference/output fails.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let voice = self.voice.unwrap_or_else(|| "p2".to_string());
        let volume = validate_volume(self.volume.unwrap_or(1.0))?;
        let output = resolve_output_path(
            &self.text,
            self.output_dir.as_deref(),
            self.output.as_deref(),
        )?;
        let alpha = self.alpha.unwrap_or(1.0);
        let (_model, runtime) = load_runtime(model_id, self.backend.as_deref())?;
        let wav = synthesize_to_wav(&runtime, &self.text, self.phonemes, &voice, alpha, volume)?;
        if let Some(output) = output.as_deref() {
            write_wav_output(&runtime, output, &wav)?;
            emit_output_path(output)?;
            tracing::info!(output = %output.display(), "playing WAV output");
        } else {
            tracing::info!("playing in-memory WAV output");
        }
        let playback_started = Instant::now();
        audio::play_wav_bytes(&wav)?;
        tracing::info!(
            elapsed_ms = playback_started.elapsed().as_millis(),
            "WAV playback complete"
        );
        Ok(CliOutput::none())
    }
}

impl WriteArgs {
    /// # Errors
    ///
    /// Returns an error if the selected model is unknown or unprepared, the
    /// text cannot be phonemized, or inference/output fails.
    #[expect(
        clippy::unused_async,
        reason = "Command invoke methods share the async CLI dispatch shape."
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_id = self.model.as_deref().unwrap_or("glados");
        let voice = self.voice.unwrap_or_else(|| "p2".to_string());
        let output = resolve_output_path(
            &self.text,
            self.output_dir.as_deref(),
            self.output.as_deref(),
        )?;
        let Some(output) = output else {
            bail!("write requires --output <path> or --output-dir <directory>");
        };
        let alpha = self.alpha.unwrap_or(1.0);
        let (_model, runtime) = load_runtime(model_id, self.backend.as_deref())?;
        let wav = synthesize_to_wav(&runtime, &self.text, self.phonemes, &voice, alpha, 1.0)?;
        write_wav_output(&runtime, &output, &wav)?;
        emit_output_path(&output)
    }
}

/// Resolve and load a prepared model for a synthesis-oriented command.
pub(crate) fn load_runtime(
    model_id: &str,
    backend: Option<&str>,
) -> Result<(model_registry::ModelDefinition, GladosRuntime)> {
    crate::config::effective_backend(backend)?;
    let Some(model) = model_registry::find_model(model_id) else {
        bail!("unknown model {model_id:?}; known models: glados");
    };
    let started = Instant::now();
    tracing::info!(model = %model.id, "loading prepared model");
    let prepared = model_registry::inspect_prepared_model_dir(model).wrap_err_with(|| {
        format!(
            "model {:?} is not prepared at the required location; {}",
            model.id,
            model_preparation_hint(model)
        )
    })?;
    let Some(model_dir) = crate::config::effective_torch_model_dir()? else {
        bail!(
            "tch/LibTorch model directory is not configured; set it once with `teamy-tts config set --torch-model-dir <path>`"
        );
    };
    let runtime = GladosRuntime::from_prepared(&prepared, &model_dir)?;
    tracing::info!(
        backend = %runtime.backend_kind(),
        elapsed_ms = started.elapsed().as_millis(),
        "prepared model loaded"
    );
    Ok((model, runtime))
}

/// Synthesize and encode one WAV while retaining the timing logs used by the
/// say, write, and interactive commands.
pub(crate) fn synthesize_to_wav(
    runtime: &GladosRuntime,
    text: &str,
    phonemes: bool,
    voice: &str,
    alpha: f32,
    volume: f32,
) -> Result<Vec<u8>> {
    let volume = validate_volume(volume)?;
    tracing::info!(voice = %voice, "synthesizing text");
    let synthesis_started = Instant::now();
    let samples = if phonemes {
        runtime.synthesize_phonemes(text, voice, alpha)?
    } else {
        runtime.synthesize(text, voice, alpha)?
    };
    let samples = scale_samples(samples, volume);
    tracing::info!(
        elapsed_ms = synthesis_started.elapsed().as_millis(),
        sample_count = samples.len(),
        volume,
        "synthesis complete"
    );
    runtime.wav_bytes(&samples)
}

fn validate_volume(volume: f32) -> Result<f32> {
    if !volume.is_finite() || !(0.0..=1.0).contains(&volume) {
        bail!("--volume must be a finite number in the range 0.0..=1.0");
    }
    Ok(volume)
}

fn scale_samples(mut samples: Vec<f32>, volume: f32) -> Vec<f32> {
    for sample in &mut samples {
        *sample *= volume;
    }
    samples
}

pub(crate) fn write_wav_output(runtime: &GladosRuntime, output: &Path, wav: &[u8]) -> Result<()> {
    tracing::info!(output = %output.display(), "writing WAV output");
    runtime.write_wav_bytes(output, wav)
}

pub(crate) fn emit_output_path(output: &Path) -> Result<CliOutput> {
    println!("{}", output.display());
    std::io::stdout()
        .flush()
        .wrap_err("failed to flush written audio path")?;
    Ok(CliOutput::none())
}

pub(crate) fn resolve_output_path(
    text: &str,
    output_dir: Option<&str>,
    output: Option<&str>,
) -> Result<Option<PathBuf>> {
    let output = match (output_dir, output) {
        (Some(directory), Some(filename)) => {
            if directory.trim().is_empty() {
                bail!("--output-dir cannot be empty");
            }
            if filename.trim().is_empty() {
                bail!("--output cannot be empty");
            }
            let filename = Path::new(filename);
            if filename.is_absolute() {
                bail!("--output must be relative when --output-dir is provided");
            }
            PathBuf::from(directory).join(filename)
        }
        (Some(directory), None) => {
            if directory.trim().is_empty() {
                bail!("--output-dir cannot be empty");
            }
            let directory = PathBuf::from(directory);
            let sequence = next_output_sequence(&directory)?;
            directory.join(format!("{sequence:04} {}.wav", sanitized_output_stem(text)))
        }
        (None, Some(filename)) => {
            if filename.trim().is_empty() {
                bail!("--output cannot be empty");
            }
            PathBuf::from(filename)
        }
        (None, None) => return Ok(None),
    };

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create output directory {}", parent.display()))?;
    }
    Ok(Some(output))
}

fn next_output_sequence(directory: &Path) -> Result<u32> {
    if !directory.exists() {
        return Ok(1);
    }
    if !directory.is_dir() {
        bail!(
            "output directory is not a directory: {}",
            directory.display()
        );
    }

    let mut next = 1_u32;
    for entry in fs::read_dir(directory)
        .wrap_err_with(|| format!("failed to inspect output directory {}", directory.display()))?
    {
        let entry = entry.wrap_err("failed to inspect an output directory entry")?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some((prefix, _)) = name.split_once(' ') else {
            continue;
        };
        let Ok(sequence) = prefix.parse::<u32>() else {
            continue;
        };
        next = next.max(
            sequence
                .checked_add(1)
                .ok_or_else(|| eyre::eyre!("output sequence number overflowed"))?,
        );
    }
    Ok(next)
}

fn sanitized_output_stem(text: &str) -> String {
    let mut stem = text
        .chars()
        .map(|character| {
            if character.is_control()
                || matches!(
                    character,
                    '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
                )
            {
                '_'
            } else {
                character
            }
        })
        .collect::<String>();
    stem.truncate(
        stem.char_indices()
            .nth(MAX_DEFAULT_OUTPUT_STEM_CHARS)
            .map_or(stem.len(), |(index, _)| index),
    );
    let stem = stem.trim().trim_end_matches([' ', '.']).to_string();
    if stem.is_empty() {
        "untitled".to_string()
    } else {
        stem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_uses_sequence_and_text() {
        let directory =
            std::env::temp_dir().join(format!("teamy-tts-output-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("0001 Existing.wav"), []).unwrap();

        let path = resolve_output_path("Hello, friend", Some(directory.to_str().unwrap()), None)
            .expect("default output should resolve");
        assert_eq!(path, Some(directory.join("0002 Hello, friend.wav")));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn explicit_output_is_joined_to_output_directory() {
        let path = resolve_output_path("ignored", Some("outputs"), Some("abc.wav"))
            .expect("explicit output should resolve");
        assert_eq!(path, Some(PathBuf::from("outputs").join("abc.wav")));
    }

    #[test]
    fn no_output_destination_is_ephemeral() {
        let path = resolve_output_path("Hello, friend", None, None)
            .expect("missing output destination should be allowed");
        assert_eq!(path, None);
    }

    #[test]
    fn volume_zero_silences_samples_without_skipping_the_buffer() {
        let samples = scale_samples(vec![-1.0, -0.25, 0.5, 1.0], 0.0);
        assert_eq!(samples, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn volume_must_be_finite_and_within_unit_range() {
        let _ = validate_volume(0.0).unwrap();
        let _ = validate_volume(1.0).unwrap();
        let _ = validate_volume(-0.01).unwrap_err();
        let _ = validate_volume(1.01).unwrap_err();
        let _ = validate_volume(f32::NAN).unwrap_err();
    }
}
