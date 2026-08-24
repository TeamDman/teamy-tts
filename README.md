# teamy-tts

`teamy-tts` is a local Rust CLI for running the GLaDOS text-to-speech models
through one native inference path: Rust bindings from `tch-rs` over LibTorch.
It does not require Python at runtime.

## Runtime model

The product runtime is intentionally narrow:

```text
teamy-tts (Rust)
    -> tch 0.24.x
    -> matching LibTorch 2.11.x CUDA runtime
    -> GLaDOS TorchScript acoustic model and vocoder
```

Burn, CubeCL, Ash/Vulkan, WGPU, and the handwritten C++ bridge remain in the
`backend-comparison` history branch. They are not part of the `main` build.
The supported build target is the MSVC Rust toolchain on Windows. The final
release package must ship the matching LibTorch DLLs beside the executable;
Python and a Python Torch installation are not required.

## Common commands

After installing the model bundle and configuring the upstream TorchScript
directory (or an equivalent packaged model directory):

```powershell
teamy-tts model list
teamy-tts config set --torch-model-dir 'C:\Models\teamy-tts\glados'
teamy-tts config set --torch-device 0

# Write and print a WAV path. A destination is required for `write`.
teamy-tts write "Hello, friend" --output .\hello.wav

# Play without creating a persistent WAV file.
teamy-tts say "Hello, friend"

# Run the complete synthesis and playback path silently.
teamy-tts say "Hello, friend" --volume 0

# Play a direct GLaDOS IPA-like phoneme sequence.
teamy-tts say --phonemes "eɪ"

# Inspect the phonemes and model token IDs produced for ordinary text.
teamy-tts phonemize "The letter A"
teamy-tts --output-format json phonemize "The letter A"

# Keep the model resident while reading lines from stdin and playing results.
# Files are retained only when --output-dir is supplied.
teamy-tts interactive
teamy-tts interactive --volume 0

# Produce JSON benchmark evidence without creating or playing output files.
teamy-tts benchmark "Hello, friend" --warmups 2 --measurements 5

# Diagnose the local installation without changing it.
teamy-tts doctor --offline
teamy-tts --output-format json doctor --deep
```

`say` and `interactive` play the generated PCM16 WAV directly from memory and
do not create files by default. Use `--output` with `say`, or `--output-dir`
with `interactive`, to retain audio. `write` always requires `--output` or
`--output-dir`; it never invents an `outputs` directory. The written path is
emitted on stdout; structured tracing remains on stderr.

`say` and `interactive` accept `--volume <0.0..=1.0>`. The multiplier is
applied to the generated PCM samples before WAV encoding and playback, so
`--volume 0` still exercises synthesis, WAV construction, and synchronous
playback while producing silence.

The `--phonemes` flag bypasses English normalization and the neural
phonemizer. Its input must use symbols from GLaDOS's IPA-like inventory, for
example `eɪ` for the spoken letter A or `hɛloʊ` for “hello”. Unsupported
symbols are rejected before inference.

`phonemize` uses the same prepared dictionary and neural phonemizer as
synthesis, but loads only the text frontend. It reports the resulting phoneme
sequence and integer token IDs without generating audio. This is useful for
debugging cases such as `A` being interpreted as `ə`; use `eɪ` with
`--phonemes` when the intended pronunciation is the name of the letter.

`doctor` reports configuration and precedence, model-cache and manifest
health, the external TorchScript directory, LibTorch/CUDA capability, audio
support, and public model-server reachability. It performs no repair and does
not modify configuration, model files, or output files. The default shallow
check avoids loading the large models; `--deep` verifies artifact hashes,
loads the actual runtime, and runs an in-memory synthesis smoke test.

Use `--offline` to skip network probes. The report has a versioned typed JSON
shape with stable check IDs, `pass`/`warn`/`fail`/`skip` statuses, evidence,
and suggested next commands. It never includes access tokens or credential
values. `--output-format text`, `json`, and `csv` are global options; CSV is a
flat one-row-per-check projection of the same diagnostic facts.

The process exits successfully when the diagnostic report itself was produced;
individual health failures are represented by the report's aggregate `status`
and check statuses so redirected JSON remains clean and useful to scripts or
an LLM.

## Model acquisition and preparation

The model catalog separates distributor (`Teamy`) from model (`glados`):

```powershell
teamy-tts model acquire-prepared Teamy
teamy-tts model prepare glados
```

The prepared bundle contains the six root-level runtime artifacts:

```text
glados-new.pt
vocoder-gpu.pt
glados-phonemizer.pt
frontend.tsv
voice-p1.f32le
voice-p2.f32le
```

Every archive and prepared artifact is verified by content hash before it is
installed. During development, a bundle directory can be prepared directly:

```powershell
teamy-tts model prepare glados --source-dir .\artifacts\glados-native
```

The raw upstream archive is retained as a separate acquisition path for model
conversion work; it is not loaded by the product at runtime.

## Building from source

Install the MSVC Rust toolchain and provision the LibTorch package matching
the pinned `tch` release. Set `LIBTORCH` only for the build, then run:

```powershell
$env:LIBTORCH = 'C:\path\to\libtorch'
cargo build --release
cargo test --all-targets
```

`update.ps1` builds and installs the executable and copies the LibTorch DLLs
beside it. It does not modify the application configuration. Set the
TorchScript model directory once, when needed:

```powershell
teamy-tts config set --torch-model-dir 'C:\Models\teamy-tts\glados'
```

Future updates preserve that remembered configuration.

## Local distribution rehearsal

The repository includes a non-publishing clean-machine rehearsal. It stages
the executable and adjacent LibTorch/CUDA DLLs, prepares the local native
bundle into an empty cache, clears inherited environment variables for child
processes, and records typed doctor, benchmark, playback, output, and failure
evidence in a versioned JSON receipt:

```powershell
.\tools\rehearse-distribution.ps1 `
  -LibTorchRoot 'G:\Programming\Caches\teamy-tts-libtorch-2.11.0-cu128\libtorch'
```

The rehearsal uses only local archives and does not contact Cloudflare,
Terraform, DNS, credentials, or a remote model server. It does not establish
rights to redistribute the model or GLaDOS voice; public publication remains a
separate authorized step.

## Benchmark status

The benchmark command reports model-load time, warmup count, sorted measured
latencies, median, p95, sample count, generated audio duration, and an explicit
correctness gate. A local probe using `tch 0.24.0` and LibTorch 2.11.0+cu128
passed the finite/stable waveform gate and canonical sample-count check on the
RTX 4090:

| Runtime | Workload | Median | P95 | Result |
|---|---|---:|---:|---|
| tch 0.24.0 / LibTorch 2.11.0+cu128 CUDA | `Hello, friend` | 57 ms | 62 ms | correctness-gated RTX 4090 receipt |

The benchmark reported a 2,590 ms model load; the generated audio contained
26,880 samples (1,219 ms). The gate rejects empty or non-finite samples and
unstable output lengths, and checks the canonical sample count for this exact
workload. A Windows MSVC CUDA-link anchor is included because the published
tch-rs 0.24 line can otherwise have the linker drop the `torch_cuda.dll`
import. The benchmark must still be repeated after the new tch-native bundle
is rehosted and acquired from a clean cache.

See [DEPENDENCIES.md](DEPENDENCIES.md) for the dependency decision and
[PLAN.md](PLAN.md) for the resumable implementation ledger.

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
