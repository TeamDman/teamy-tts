# Native GLaDOS inference

This crate runs the complete GLaDOS model from Rust and ahead-of-time CUDA code. The runtime loads numerical tensors and a text dictionary. It does not load TorchScript, LibTorch, Python or a serialized computation graph.

The model includes the six-layer DeepPhonemizer transformer, MultiForwardTacotron acoustic model and HiFiGAN vocoder. Rust defines layer order, dimensions, duration expansion and speaker selection. CUDA kernels implement the tensor operations; cuBLAS supplies matrix multiplication and cuDNN 9 supplies bidirectional GRU and LSTM inference.

The current native backend targets NVIDIA CUDA. The existing default `tch-native` build remains available, including its CPU option. Both voices, duration scaling, phoneme input, WAV output and the existing interactive playback loop use the native backend when built with `cuda-native`.

## Build

Install a CUDA toolkit and a compatible cuDNN 9 runtime. Set `CUDA_PATH` to the toolkit. The build defaults to `sm_89`; set `GLADOS_CUDA_ARCH` to your GPU's supported architecture before compiling for another device.

From the repository root:

```powershell
cargo build --release --no-default-features --features cuda-native --target-dir target/native-cli
```

The native CUDA runtime currently selects device 0. Put the CUDA runtime and cuBLAS DLL directories on `PATH`. Put cuDNN's runtime directory on `PATH`, or set `GLADOS_CUDNN_LIBRARY` to its cuDNN 9 library. cuDNN's dependent libraries must also be available. These are vendor runtime libraries; the native executable does not need Torch DLLs.

## Prepare the numerical artifact

Use a development Python environment with PyTorch and NumPy to export an existing prepared GLaDOS model directory:

```powershell
python native/tools/export_model.py --models-dir /path/to/prepared/glados-new --output artifacts/native-glados
```

The source directory must contain `glados-new.pt`, `vocoder-gpu.pt`, `glados-phonemizer.pt`, `frontend.tsv` and both `voice-p*.f32le` files. The exporter validates the deployed frozen vocoder's constants and records source hashes. The resulting `weights.safetensors` contains 614 numerical tensors, including both voices. Large files stay under the ignored `artifacts` directory.

The runtime needs only `weights.safetensors` and `frontend.tsv`. Keep `manifest.json` with distributed weights to retain conversion provenance. Other exporter outputs are local validation fixtures and audit material, not runtime requirements.

## Use the existing CLI

Set `TEAMY_TTS_NATIVE_MODEL_DIR` to the exported artifact directory. If an existing configuration explicitly selects LibTorch, select `cuda-native` with `TEAMY_TTS_BACKEND` or the configuration command below. Then run:

```powershell
target/native-cli/release/teamy-tts.exe phonemize "Hello, friend"
target/native-cli/release/teamy-tts.exe write "Hello, friend" --output hello.wav
target/native-cli/release/teamy-tts.exe interactive
target/native-cli/release/teamy-tts.exe benchmark "Hello, friend" --warmups 3 --measurements 20
target/native-cli/release/teamy-tts.exe doctor --offline --deep
```

To remember the directory and backend, use `config set --backend cuda-native --native-model-dir <directory>`. Environment variables override remembered settings. The same configuration file is shared with the default build unless `TEAMY_TTS_HOME_DIR` selects a separate home.

Native startup warms the neural frontend and complete synthesis path before the interactive ready message. Novel sequence lengths still incur their own allocation and execution costs. Interactive mode keeps model weights resident and exits on EOF or cancellation.

The CUDA allocator uses a 512 MiB retention threshold between requests. Set `GLADOS_CUDA_POOL_LIMIT_MB` from 0 to 4096 to compare memory and latency. This controls retained pool memory, not total GPU memory or a hard allocation limit.

Persistent cuDNN recurrent algorithms are available only with the `experimental-rnn` development feature. They failed the strict full-waveform gate and are not enabled in the normal native build.

## Validate and profile

Build the development stage benchmark executable separately:

```powershell
cargo build --release --manifest-path native/Cargo.toml
cargo test --release --manifest-path native/Cargo.toml --lib
cargo test --release --manifest-path native/Cargo.toml --lib -- --ignored --test-threads=1
```

The ignored tests require CUDA. They compare convolutions, transpose convolutions, attention, normalization, pooling and indexing against independent double-precision CPU calculations, including short and uneven shapes.

Development tools:

| Tool | Purpose |
|---|---|
| `export_phonemizer_fixtures.py` | Independent upstream transformer logits and exact predicted IDs |
| `export_text_fixture.py` | Acoustic intermediates and PCM for a real text case |
| `compare_cli.py` | Serial release comparisons across voices, rates, rare words, numbers and longer input |
| `benchmark_python.py` | Actual upstream Python API and matched vocoder timing |
| `benchmark_python_launch.py` | External Python process launch to readiness and first complete host PCM |
| `benchmark_interactive.py` | External launch-to-ready, silent real playback, host memory and loaded modules |

The waveform gate requires exact sample count, finite output, relative RMS error at most 0.001 and peak absolute error at most 0.001 against the deployed upstream FP32 reference. Neural frontend decisions must match exactly. Timings alone do not pass correctness.

Keep default product precision and strict FP32 product results separate. `NVIDIA_TF32_OVERRIDE=0` disables TF32 for the comparison process. The current product's default CUDA output can differ substantially from the upstream FP32 waveform. Even strict FP32 results can differ between Torch and cuDNN versions; retain the common reference and report each error instead of silently changing tolerances.

Benchmark receipts must identify binary hashes, source revision, runtime versions, warmups, raw samples and timing boundaries. Launch-to-ready, first output and resident inference are separate measurements. Warm filesystem caches do not establish storage-cold startup. The interactive probe measures handoff to the audio API, not physical speaker latency.
