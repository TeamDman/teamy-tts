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

# Write and print a WAV path.
teamy-tts write "Hello, friend"

# Write, print, and play a WAV.
teamy-tts say "Hello, friend"

# Keep the model resident while reading lines from stdin and playing results.
teamy-tts interactive

# Produce JSON benchmark evidence without creating or playing output files.
teamy-tts benchmark "Hello, friend" --warmups 2 --measurements 5
```

`write` defaults to `outputs/0001 <text>.wav`. Use `--output-dir` for an
automatic numbered-output directory or `--output` for an explicit filename.
The written path is emitted on stdout; structured tracing remains on stderr.

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

`update.ps1` builds and installs the executable, copies the LibTorch DLLs
beside it, and remembers the TorchScript model directory in the application
configuration so future invocations do not need environment variables.

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
