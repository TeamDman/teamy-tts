# teamy-tts

Local Rust text-to-speech CLI based on the GLaDOS TTS pipeline, with Burn,
Burn WGPU, LibTorch/TorchScript, and Vulkan candidate backends.

Target command surface:

~~~powershell
teamy-tts model list
teamy-tts config show
teamy-tts backend list
teamy-tts backend benchmark --backend burn-cuda-acoustic
teamy-tts model acquire-prepared Teamy
teamy-tts write --model glados "hello!" --output .\explicit.wav
teamy-tts say --model glados "hello!"
teamy-tts interactive --model glados --output-dir .\clips
teamy-tts model acquire-unprepared Teamy
teamy-tts model acquire-unprepared R2D2FISH-OneDrive
teamy-tts model acquire-prepared R2D2FISH-OneDrive
teamy-tts model prepare glados --source-dir <native-bundle-directory>
teamy-tts model prepare glados --source-archive <native-bundle.zip>
~~~

The executable should be downloadable and usable without a checkout of the
upstream Python repository. Model assets are verified, cached, and reported by
the CLI. The prepared bundle command installs the native model automatically;
no separate `model prepare` command is needed for the normal Teamy path.

This project is licensed under the [Mozilla Public License 2.0](LICENSE).

The raw upstream archive is a separate acquisition step. The Teamy source
will be rehosted in Cloudflare R2, while R2D2FISH-OneDrive represents the
upstream-maintainer source. Terraform owns the R2 infrastructure; publication
and post-upload verification must prove that the immutable archive exists
before the CLI catalog points at it.

The first pure-Rust inference slice is now working: the template-based
executable supports model catalog inspection, verified raw-archive acquisition,
native bundle preparation, Burnpack-backed DeepPhonemizer/ForwardTacotron/
HiFiGAN loading, and `write`/`say` WAV output. `interactive` keeps the prepared
model loaded while it reads, writes, and plays one line at a time. Raw upstream
TorchScript conversion is still a development-time step, and full
cleaner/reference parity plus clean-device packaging remain release gates. The
plain CUDA build also contains a backend-native `CubeCL` LSTM kernel; the
portable packed Burn implementation remains the fallback for fused and
all-backends builds. The
executable plan is in
[PLAN.md](PLAN.md).

Cloudflare R2 infrastructure is defined in
[infra/cloudflare](infra/cloudflare/README.md).

## Current Rust CLI slice

```powershell
cargo run -- model list
cargo run -- --output-format json model show glados

# The native Teamy bundle also uses its baked R2 URL by default. This command
# downloads, verifies, and installs the bundle in one step.
# TEAMY_TTS_TEAMY_NATIVE_SOURCE_URL may override it for a test mirror.
cargo run -- model acquire-prepared Teamy

# The raw archive is for converter/developer workflows; it is not ready for
# `say` until a native bundle is prepared.
# TEAMY_TTS_TEAMY_SOURCE_URL may override the baked raw URL for a test mirror.
cargo run -- model acquire-unprepared Teamy

# Install a converter-produced native bundle directory for local inference.
cargo run -- model prepare glados --source-dir .\artifacts\native-bundle

# Or install the six-artifact ZIP directly; extraction is performed in Rust.
cargo run -- model prepare glados --source-archive .\artifacts\teamy-tts-glados-new-native-bundle.zip

# The default build enables the CUDA vocoder path when the CUDA toolkit/driver
# are available on the build machine.
cargo run --release -- say --model glados "hello!"

# Force the current Burn CPU-acoustic/CUDA-vocoder candidate.
cargo run --release -- say --model glados --backend burn "hello!"

# Benchmark the Burn candidates on an RTX 4090.
cargo run --release -- backend benchmark --backend burn-ndarray
cargo run --release -- backend benchmark --backend burn-cuda-acoustic

# The plain CUDA build enables the backend-native CubeCL recurrent kernel.
# `all-backends` remains the portable comparison distribution and uses the
# packed recurrent fallback when Burn fusion/WGPU/Vulkan change the CUDA alias.
cargo run --no-default-features --features cuda --release -- `
  backend benchmark --backend burn-cuda-acoustic

# Build and benchmark Burn fusion/autotune as a separate candidate.
cargo run --no-default-features --features burn-cuda-fused --release -- `
  backend benchmark --backend burn-cuda-fused

# Build and benchmark Burn's WGPU backend with automatic graphics API selection.
cargo run --no-default-features --features burn-wgpu --release -- `
  backend benchmark --backend burn-wgpu

# Build and benchmark Burn's explicit Vulkan/SPIR-V backend. The Cargo patch
# pins the matching CubeCL v0.8.1 sources because cubecl-spirv is not on crates.io.
cargo run --no-default-features --features burn-vulkan --release -- `
  backend benchmark --backend burn-vulkan

# Burn tch is a separate LibTorch ABI candidate. Point the build at a matching
# PyTorch 2.9 installation (the upstream PyTorch 2.0 environment is not enough).
$env:LIBTORCH = 'G:\Programming\Repos\teamy-tts\target\teamy-tts-pytorch-29\Lib\site-packages\torch'
$env:LIBTORCH_USE_PYTORCH = '1'
$env:Path = 'G:\Programming\Repos\teamy-tts\target\teamy-tts-pytorch-29\Scripts;' + "$env:LIBTORCH\lib;$env:Path"
cargo run --no-default-features --features burn-tch --release -- `
  backend benchmark --backend burn-tch

# Write a WAV without playing it.
cargo run --release -- write --model glados "hello!" --output .\explicit.wav

# Keep the model warm while reading lines from stdin; EOF exits.
cargo run --release -- interactive --model glados --output-dir .\clips

# Portable fallback: both neural graphs run on the Burn CPU backend.
cargo run --no-default-features --release -- say --model glados "hello!"

# Optional fast local CUDA path. This uses the upstream TorchScript files
# through a LibTorch C++ bridge and keeps the model warm for interactive use.
$env:LIBTORCH = 'G:\ml\glados-tts-upstream\.venv\Lib\site-packages\torch'
$env:Path = "$env:LIBTORCH\lib;$env:Path"
$env:TEAMY_TTS_TORCH_MODEL_DIR = 'G:\ml\glados-tts-upstream\models'
cargo run --features torchscript --release -- interactive --model glados

# Force the Python-free LibTorch/TorchScript candidate.
cargo run --features torchscript --release -- write `
  --model glados --backend libtorch "hello!" --output .\libtorch.wav

# Build the ordinary inference backends (Burn/CUDA, Burn WGPU, LibTorch, and Vulkan).
cargo build --release --features all-backends
```

`all-backends` is the buildable distribution bundle for the current upstream
toolchain. Burn's `tch` candidate is intentionally opt-in because it requires
the newer LibTorch version selected by `tch` 0.22; build it separately with
`--features burn-tch` after provisioning that matching runtime.

The opt-in Burn-tch path specializes the GRU, CBHG, and LSTM boundaries with
LibTorch tensor operations while retaining Burn-owned modules and Burnpack
artifacts. Its current follow-up is persistent contiguous cuDNN parameter
storage, which should remove the repeated weight-repacking warning and reduce
the long-form latency tail.

The repository's `update.ps1` is the normal Windows installation bootstrap.
It builds with `all-backends`, uses `LIBTORCH` only during compilation, copies
the matching LibTorch DLLs beside the installed executable, and remembers the
TorchScript model directory when `-TorchModelDir` (or the one-time
`TEAMY_TTS_TORCH_MODEL_DIR` override) is supplied. On this development
machine it also discovers the known upstream LibTorch and model paths:

~~~powershell
.\update.ps1
teamy-tts config show
teamy-tts interactive --backend libtorch
~~~

Runtime settings can be changed through the CLI and are stored under the
application home in `config.json`:

~~~powershell
teamy-tts config set --backend libtorch `
  --torch-model-dir 'G:\ml\glados-tts-upstream\models' `
  --torch-device 0
teamy-tts config set --model-dir 'D:\teamy-tts-models'
teamy-tts config show
teamy-tts config clear --torch-device
~~~

Explicit command-line values take priority, followed by environment overrides,
then remembered configuration, then built-in defaults. The supported runtime
overrides are `TEAMY_TTS_BACKEND`, `TEAMY_TTS_MODEL_DIR`,
`TEAMY_TTS_TORCH_MODEL_DIR`, and `TEAMY_TTS_TORCH_DEVICE`; they are useful for
temporary experiments and are not required for ordinary installed use.

With no output flags, `say` creates the `outputs` directory and chooses the
next numbered filename from the text, for example
`outputs\0001 Hello, friend.wav`. `--output-dir` changes the directory and
keeps automatic numbering; adding `--output greeting.wav` selects a filename
inside that directory. Without `--output-dir`, `--output` is treated as the
complete output path. Progress and timing logs are written to stderr. Each
generated audio command prints only the written WAV path to stdout; there is no
structured stdout report. `say` prints the path after writing and
then plays the file synchronously; `write` stops after writing; `interactive`
reuses the loaded model for each non-empty stdin line, prints each path, and
plays each file. Tracing progress and timing logs continue on stderr.
Its stdin wait is cancellation-aware: Ctrl+C is selected alongside the reader,
so a line arriving concurrently with cancellation is discarded instead of being
synthesized, and a blocked stdin reader cannot keep the command waiting during
shutdown. EOF still joins the reader normally.
In the default build, `auto` uses Burn until a correctness-passing benchmark
receipt exists. The named Burn candidates are `burn-ndarray` (CPU control),
`burn` (CPU acoustic graph plus CUDA vocoder when CUDA is enabled),
`burn-cuda-acoustic` (both neural graphs on CUDA), `burn-cuda-fused` (CUDA
fusion/autotune), `burn-tch` (Burn's LibTorch/tch backend), `burn-wgpu`
(Burn's WGPU backend with automatic graphics API selection), and
`burn-vulkan` (Burn's explicit Vulkan backend). Fusion, tch, WGPU, and
Burn Vulkan are separate build candidates; each must pass the same waveform gate
before it can be considered. `burn-tch` additionally requires a LibTorch
installation compatible with the `tch` version selected by Burn. When the
optional `torchscript` feature is enabled, `auto` may select LibTorch only after
`backend benchmark --backend libtorch` writes a receipt matching the prepared
model, GPU/driver, executable revision, and benchmark configuration. Stale,
malformed, failing, or unavailable receipts are ignored and Burn is used.
`cargo run --release` is the recommended path when synthesizing longer text.

The optional `torchscript` feature is the fast local development path. It
requires a matching LibTorch/PyTorch installation at build and run time, and
the configured TorchScript model directory must contain `glados-new.pt` and
`vocoder-gpu.pt`. It follows the upstream device split: the acoustic graph runs
on CPU and HiFiGAN runs on CUDA. The runtime performs two warmup passes while
loading the model; after that, local measurements were about 74 ms for
`Hello, friend` and 221 ms for `Let's see how fast this goes.` on the inspected
RTX 4090. The first command therefore has a slower model-load phase, while
`interactive` reuses the warmed model. Without a passing benchmark receipt,
`--backend auto` uses Burn; with
`--backend libtorch`, the same condition is an actionable error. The
`--backend vulkan` is available in a `vulkan` build as an experimental
RTX-4090-targeted backend: it dispatches the prepared acoustic embedding,
fixed-shape pitch/energy condition projections, a four-gate forward/reverse
LSTM candidate, the CBHG postnet, acoustic mel/post projection, and the complete
HiFi-GAN vocoder through a persistent Ash context. Every migrated intermediate
can be parity-checked against the Burn graph with
`TEAMY_TTS_VULKAN_PARITY=1`; normal execution keeps only the Burn
predictor/prenet prefix and does not compute the duplicate recurrent or
postnet graph. This is still not an upstream-class performance claim because
the predictor/prenet prefix and host-visible staging remain. Model weights
persist and each Vulkan graph is submitted as one device-resident batch. `auto`
only selects it when a passing benchmark receipt proves it is the best available
candidate.

For a supported Windows LibTorch accelerator package, build the
`torchscript` feature with the intended LibTorch installation and copy the
executable plus its native DLLs into a separate staging directory:

~~~powershell
& .\tools\package-libtorch-runtime.ps1 -ExecutablePath .\target\release\teamy-tts.exe -LibTorchRoot $env:LIBTORCH -OutputDir .\artifacts\teamy-tts-libtorch
~~~

The packager copies every DLL from `LibTorchRoot\lib` except
`torch_python.dll` by default, writes `libtorch-runtime.json` with SHA-256
values, and does not copy or invoke Python. The GLaDOS model bundle remains a
separate verified artifact because it is large and has separate redistribution
provenance. A release rehearsal must place that model bundle beside the
staging directory, set `TEAMY_TTS_TORCH_MODEL_DIR` to its TorchScript files,
put the staging directory first on `PATH`, and run repeated `write` commands
from a shell with no upstream Python checkout on `PATH`. The optional
`-IncludeTorchPython` switch exists only for diagnosing an installation whose
LibTorch build unexpectedly requires that DLL; it is not part of the supported
Python-free package.

The Vulkan candidate is packaged as a feature-specific executable because its
SPIR-V shaders are embedded at build time and it has no extra native DLL set:

~~~powershell
cargo build --release --no-default-features --features vulkan
& .\tools\package-vulkan-runtime.ps1 `
  -ExecutablePath .\target\release\teamy-tts.exe `
  -OutputDir .\artifacts\teamy-tts-vulkan
~~~

The resulting `vulkan-runtime.json` records the executable hash and the
external Vulkan-driver requirement. The validated target is the NVIDIA RTX
4090; the verified model bundle remains a separate artifact.

Backend evidence can be inspected and generated with:

~~~powershell
cargo run --release -- backend list
cargo run --release -- backend benchmark --backend burn-ndarray
cargo run --release -- backend benchmark --backend burn-cuda-acoustic
cargo run --release -- backend benchmark --backend burn-cuda-acoustic `
  --corpus glados-long-v1 --warmup 1 --measurements 1 --skip-correctness
cargo run --no-default-features --features burn-cuda-fused --release -- `
  backend benchmark --backend burn-cuda-fused
cargo run --release --features torchscript -- backend benchmark --backend libtorch
cargo run --no-default-features --features vulkan -- backend probe
cargo run --no-default-features --features vulkan -- write --backend vulkan --model glados "hello!" --output .\vulkan.wav
~~~

The Vulkan probe is an RTX 4090-targeted substrate check. It validates
vector-add, fixed 16x16 matrix-multiply, and a persistent model-shaped
embedding lookup against CPU results, reports cooperative-matrix capabilities,
and reports GPU timestamp duration when the queue exposes timestamp queries.
The experimental Vulkan runtime uses that persistent context for the actual
prepared acoustic embedding, condition projections, fixed-shape LSTM/mel
projection candidate, CBHG postnet, acoustic post projection, and all HiFi-GAN
convolution stages, with each result gated against the Burn reference by the
shared benchmark command. The predictor/prenet remain Burn-backed; setting
`TEAMY_TTS_VULKAN_PARITY=1` additionally runs the complete Burn acoustic graph
for intermediate diagnostics. The Ash batch keeps model weights and
intermediate tensors in device-local memory between dispatches, using
host-visible staging only for batch inputs and final readback;
`TEAMY_TTS_VULKAN_PARITY=1 --debug` enables stage-level diagnostics against
the Burn graph. The probe
reports separate cold and warm dispatch timings so shader/pipeline setup is
not confused with reusable inference work.

An earlier full matrix run on the inspected RTX 4090 used the
`glados-short-v1` two-text corpus and measured the candidates before the
packed recurrent checkpoint:

| Candidate | Warm median | Correctness | Notes |
|---|---:|---|---|
| `burn-ndarray` | 40,491 ms | pass | CPU reference |
| `burn` | 2,080 ms | control | CPU acoustic plus CUDA vocoder |
| `burn-cuda-acoustic` | 339 ms | pass | Both neural graphs on Burn CUDA |
| `burn-cuda-fused` | 3,967 ms | fail | Historical pre-fix result: 41,216 vs 40,960 samples |
| `burn-wgpu` | 4,267 ms | pass | WGPU automatic graphics API selection |
| `burn-vulkan` | 4,759 ms | pass | Burn explicit Vulkan/SPIR-V; first-use kernel compilation included |
| `libtorch` | 340 ms | pass | Direct upstream TorchScript through native LibTorch |
| `vulkan` | 834 ms | pass | Specialized Ash Vulkan path |
| `burn-tch` | 492 ms | historical fail | Pre-specialization LibTorch 2.9 run; output length changed: 41,216 vs 40,960 samples |

These are single-measurement diagnostic runs with candidate-specific warmup
counts (zero or one), not release performance claims. The Burn WGPU and Burn
Vulkan receipts include first-use graphics compilation in their single
measurements. Burn tch is now compiled and measured against its matching
LibTorch 2.9 runtime; the historical output-shape mismatch was fixed at the
shared duration boundary. The Burn tch row is an explicit toolchain boundary,
not a claim that Burn and direct LibTorch are the same backend. After moving the CBHG postnet off Burn, recording the
postnet and vocoder in one Vulkan batch, and replacing per-tensor Vulkan memory
allocations with an arena, a stabilized one-warmup/three-measurement release
benchmark measured Vulkan at `1314.138 ms` median with
`relative_rms_error=0.000187` and `max_abs_error=0.000560`; both candidates
passed the shared correctness gate under their respective receipts. The merged
batch removes the intermediate host round-trip, but Vulkan remains an explicit
experimental override while its allocation, dispatch, and latency costs are
profiled; `backend list` and `--backend auto` select LibTorch when its
Python-free native runtime and model directory are available.

The packed-gate Burn GRU/LSTM checkpoint was also measured against a
representative long-form workload on the inspected RTX 4090. Plain CUDA now
adds a backend-native `CubeCL` bidirectional LSTM kernel over Burn's CUDA tensor
handles; fused/all-backends builds retain the packed fallback because Burn's
fusion, WGPU, and explicit Vulkan features change the `Cuda` primitive alias.
The Burnpack module layout remains unchanged:

| Candidate | Warm median | Output duration | Real-time factor | Correctness |
|---|---:|---:|---:|---|
| `burn-cuda-acoustic` (packed) | 2,142.106 ms | 8,591 ms | 0.2493 | short corpus pass; long comparison skipped |
| `burn-cuda-acoustic` (CubeCL, plain `cuda`) | 1,930.462 ms | 8,591 ms | 0.2247 | short corpus pass; long comparison skipped |
| `libtorch` | 371.905 ms | 8,591 ms | 0.0433 | short corpus pass; long comparison skipped |
| `burn-tch` (fused GRU/CBHG/LSTM) | 271.760 ms | 8,591 ms | 0.0333 | short corpus pass; long comparison skipped |

Both rows generated `189,440` samples. The long corpus is diagnostic because
the CPU NdArray reference takes several minutes at this frame count;
`--skip-correctness` explicitly prevents its receipt from influencing
`--backend auto`. Both Burn paths retain the original Burnpack module layout;
the packed path combines compatible recurrent gate projections, while the
plain-CUDA path computes the LSTM recurrence in one `CubeCL` launch per
direction. The CubeCL first-use compilation is excluded by the benchmark
warmup, so this is a warm performance checkpoint rather than a new model
format.

The duration-shape drift in the fused candidate is corrected at the shared
frame conversion boundary. Values within `0.004` of a half-frame use
PyTorch-style tie-to-even rounding; the affected fused item now produces 160
frames and 40,960 samples, matching the NdArray reference. Its refreshed
short-corpus receipt passes with `relative_rms_error=0.070197` and a
`262.981 ms` warm median. This correction is separate from the remaining
kernel-launch overhead that keeps the plain Burn CUDA path behind the direct
LibTorch oracle. The Burn-tch candidate now uses LibTorch's fused GRU, CBHG
tensor, and LSTM operations while retaining the Burn module layout and runtime
contract. Its long-form receipt reports a `271.760 ms` warm median
(`316.495 ms` p95), and its short correctness receipt passes with
`relative_rms_error=0.071774`.

The refreshed one-warmup/one-measurement short-corpus checkpoint is:

| Candidate | Warm median | Correctness |
|---|---:|---|
| `burn-cuda-acoustic` (packed) | 270.525 ms | pass |
| `burn-cuda-acoustic` (CubeCL, plain `cuda`) | 271.804 ms | pass |
| `burn-cuda-fused` | 262.981 ms | pass |
| `libtorch` | 74.186 ms | pass |

A later device-local model/batch build passed the shared correctness gate at
`1303.979 ms` median in an earlier three-sample run. After narrowing the fixed
graph's compute barriers from write-to-read/write to write-to-read, the default
two-warmup/three-measurement receipt measured Vulkan at `903.9575 ms` median
(the preceding fingerprinted receipt measured `1047.4319 ms`) with the same
correctness result. This is a repeatable improvement on the
inspected RTX 4090, but Vulkan remains experimental and substantially slower
than LibTorch; it is not eligible to replace the current automatic choice.

For Vulkan performance investigation, set `TEAMY_TTS_VULKAN_PROFILE=1`. The
experimental backend then logs host command-recording time, GPU timestamp time,
temporary tensor slices, and arena bytes for each batch; the variable is not
needed for ordinary synthesis.

Benchmark receipts are stored under the application cache in
`backend-benchmarks/<model>/`. They are evidence, not model artifacts:
a receipt is only eligible for `auto` when its correctness gate passes and
all model, device, backend-build/source-fingerprint, and workload-key fields
still match.

The repository's `.cargo/config.toml` pins the cudarc toolkit selector to
CUDA 13.0 because the installed cudarc release does not yet name CUDA 13.3;
the CUDA 13.0 selector is ABI-compatible with the installed 13.3 toolkit.
This is a non-secret build compatibility setting, not a credential.

The default distributed path remains Burn-native and does not require Python,
LibTorch, or the upstream checkout. The optional TorchScript bridge exists so
the local CUDA workflow can use PyTorch's cuDNN/JIT execution while the Burn
implementation continues to serve as the portable product path. The bridge is
implemented directly against the installed LibTorch C++ API because the
available `tch` bindings target a newer PyTorch ABI than the upstream
PyTorch 2.0.1 environment.

The native bundle contains `acoustic-model.bpk`, `vocoder.bpk`,
`phonemizer.bpk`, `frontend.tsv`, and the two `voice-*.f32le` files. The
development-only Python tools create these files from the upstream TorchScript
weights; the packaged `teamy-tts` executable does not load Python or the
upstream checkout.

Package those six files for a release or model host with:

```powershell
& .\tools\package-native-bundle.ps1 `
  -SourceDir .\artifacts\native-bundle `
  -OutputPath .\artifacts\teamy-tts-glados-new-native-bundle.zip
```

The archive is an input to `model prepare --source-archive` for local or
converter workflows. `model acquire-prepared` uses the same Rust extraction
and verification path after downloading the catalogued native archive, so an
end user does not need to run `model prepare` separately. Its checksum must be
recorded in the model source manifest before publishing it.

The development-only TorchScript inventory tool records upstream file hashes,
methods, operators, state-dictionary tensor shapes, and vocoder operator gaps:

```powershell
& 'G:\ml\glados-tts-upstream\.venv\Scripts\python.exe' `
  .\tools\inspect_glados.py G:\ml\glados-tts-upstream\models `
  --output .\artifacts\glados-inventory.json
```

The frontend parity receipt can be regenerated and checked against the pinned
upstream checkout:

```powershell
& 'G:\\ml\\glados-tts-upstream\\.venv\\Scripts\\python.exe' `
  .\\tools\\reference_glados_frontend.py `
  --upstream-root G:\\ml\\glados-tts-upstream `
  --corpus .\\reference\\frontend-corpus.json --check
```

The native phonemizer diagnostic compares the converted Burnpack at the
intermediate Transformer stages:

```powershell
cargo run --release --example verify_glados_phonemizer -- `
  .\\artifacts\\glados-phonemizer.bpk supercalifragilistic
```

The acquisition command writes a partial file first, verifies the catalogued
byte count and SHA-256, then atomically installs the archive and its receipt
under the content-addressed cache. The model root can be overridden with
`TEAMY_TTS_MODEL_DIR`; the raw archive cache follows the app cache resolved by
`TEAMY_TTS_CACHE_DIR`.

## Terraform Cloudflare session

The repository contains the non-secret 1Password item reference, Cloudflare
account ID, and Azure state backend literals. The credential values themselves
are read only through `op`:

```powershell
Set-Location .\infra\cloudflare
. ..\..\get-cloudflare-token.ps1
```

```powershell
terraform init
terraform plan
```

The loader uses `op read` and does not write secrets to repository files. It
must be dot-sourced; running it as a separate process would not preserve its
environment variables for Terraform. Clear credentials when finished if this
terminal will remain open:

```powershell
Remove-Item Env:\CLOUDFLARE_API_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:\AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
Remove-Item Env:\AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
```
