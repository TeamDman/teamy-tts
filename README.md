# teamy-tts

`teamy-tts` is a local Rust CLI for running the GLaDOS text-to-speech models.
The default backend runs source-defined CUDA kernels. An optional `tch-rs`
and LibTorch backend remains available. Neither needs Python at runtime.

## Source-defined CUDA backend

The `cuda-native` build expresses DeepPhonemizer, ForwardTacotron and HiFiGAN
in Rust and ahead-of-time CUDA kernels. It loads weights from a safetensors
artifact, with no TorchScript or LibTorch dependency. cuBLAS and cuDNN 9
provide matrix multiplication and recurrent primitives.

```powershell
cargo build --release
target/release/teamy-tts.exe config set --backend cuda-native --native-model-dir <exported-model-directory>
target/release/teamy-tts.exe interactive
```

See [native build, artifact export and validation instructions](native/README.md).
The native backend currently uses CUDA device 0. Build with
`--no-default-features --features tch-native` to use the legacy backend,
including its CPU option and model packaging flow.

## Optional LibTorch runtime

The optional `tch-native` build uses:

```text
teamy-tts (Rust)
    -> tch 0.24.x
    -> matching LibTorch 2.11.x CUDA runtime
    -> GLaDOS TorchScript acoustic model and vocoder
```

Burn, CubeCL, Ash/Vulkan, WGPU, and the handwritten C++ bridge remain in the
`backend-comparison` history branch. They are not part of the `main` build.
The supported build target is the MSVC Rust toolchain on Windows. The final
LibTorch release package must ship matching LibTorch DLLs beside the executable;
Python and a Python Torch installation are not required.

## Common commands

After installing the exported native model and vendor runtime DLLs:

```powershell
teamy-tts config set --backend cuda-native --native-model-dir 'C:\Models\teamy-tts\glados-native'

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

In `interactive`, press Ctrl-D on an empty line to exit normally. If you have
typed text, Ctrl-D submits it first; press Ctrl-D again at the empty prompt to
exit. Windows console editing still supports Backspace, Unicode input, and
Ctrl-Z followed by Enter. Redirected stdin exits when its input reaches EOF.

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

`doctor` reports configuration and precedence, backend-specific model artifact
health, CUDA or LibTorch capability, audio
support, and public model-server reachability. It performs no repair and does
not modify configuration, model files, or output files. The default shallow
check avoids loading the large models; `--deep` validates model loading
and runs an in-memory synthesis smoke test. The LibTorch path also verifies
the prepared manifest's artifact hashes.

Use `--offline` to skip network probes. The report has a versioned typed JSON
shape with stable check IDs, `pass`/`warn`/`fail`/`skip` statuses, evidence,
and suggested next commands. It never includes access tokens or credential
values. `--output-format text`, `json`, and `csv` are global options; CSV is a
flat one-row-per-check projection of the same diagnostic facts.

The process exits successfully when the diagnostic report itself was produced;
individual health failures are represented by the report's aggregate `status`
and check statuses so redirected JSON remains clean and useful to scripts or
an LLM.

## LibTorch model acquisition and preparation

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

Install the MSVC Rust toolchain, CUDA toolkit and cuDNN 9. Set `CUDA_PATH`
to the toolkit. The native build defaults to the RTX 4090's `sm_89` target;
set `GLADOS_CUDA_ARCH` before building for another GPU.

```powershell
cargo build --release
cargo test --release --all-targets
```

`update.ps1` installs the default native release and copies CUDA/cuDNN DLLs
beside it. For the first installation, supply an exported model directory
and a cuDNN runtime directory:

```powershell
.\update.ps1 -CudnnRoot 'C:\path\to\cudnn' -NativeModelDir 'C:\Models\teamy-tts\glados-native'
```

The updater copies weights into the Cargo installation's
`share/teamy-tts/glados-native-v1` directory and remembers that location.
It selects `cuda-native`, preserves other settings, and checks the installed
runtime without development DLL paths. Future `.\update.ps1` runs reuse the
installed runtime DLLs and weights. No persistent environment changes are needed.

To install the legacy backend, run
`.\update.ps1 -Backend libtorch -LibTorchRoot 'C:\path\to\libtorch'`.
That build uses `--no-default-features --features tch-native` and selects
`libtorch` in configuration.

## Legacy LibTorch distribution rehearsal

The repository includes a non-publishing clean-machine rehearsal. It stages
the executable and adjacent LibTorch/CUDA DLLs, prepares the local native
bundle into an empty cache, clears inherited environment variables for child
processes, and records typed doctor, benchmark, playback, output, and failure
evidence in a versioned JSON receipt:

```powershell
.\tools\rehearse-distribution.ps1 `
  -ExecutablePath '.\target\libtorch\release\teamy-tts.exe' `
  -LibTorchRoot 'G:\Programming\Caches\teamy-tts-libtorch-2.11.0-cu128\libtorch'
```

The rehearsal uses only local archives and does not contact Cloudflare,
Terraform, DNS, credentials, or a remote model server. It does not establish
rights to redistribute the model or GLaDOS voice; public publication remains a
separate authorized step.

## Historical LibTorch benchmark

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
