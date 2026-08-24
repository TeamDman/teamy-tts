# teamy-tts implementation plan

Status: the backend-comparison checkpoint is preserved, while `main` is now a
single tch/LibTorch runtime. The direct Rust `tch::CModule` runtime compiles
against tch 0.24/LibTorch 2.11, detects the RTX 4090, and has a
correctness-gated warm benchmark receipt. The packaged executable also passes
with adjacent native DLLs. The next implementation goal is a typed `doctor`
diagnostic surface that reuses the teamy-rust-cli output contract and helps a
consumer or LLM identify missing model, native-runtime, CUDA, audio, or model
server prerequisites without self-repairing them. The local clean-machine
distribution rehearsal and silent playback control are now repeatable; external
publication remains
 explicitly gated on model/voice redistribution rights and operator approval.
Plan owner: Teamy
Plan path: G:\Programming\Repos\teamy-tts\PLAN.md
Last updated: 2026-08-23
Current focus: [x] preserve the historical backend comparison; [x] remove
non-tch code and migrate the prepared bundle to TorchScript artifacts; [x] pin
the final single-backend Torch/CUDA toolchain; [x] add and validate the typed
`doctor` report; [x] make playback in-memory and output persistence opt-in;
[x] add direct GLaDOS phoneme input; [x] complete the local clean-machine
distribution rehearsal; [x] add bounded silent playback control to `say` and
`interactive`; [!] complete external publication after rights and operator
authorization are established.

**Plan status:** Active
**Primary implementation root:** `G:\Programming\Repos\teamy-tts`
**Intent audit:** Passed 2026-08-12 for the doctor-surface slice against the
available current guidance and the durable ledger below. Historical details
from earlier compacted conversation are represented by T1-T30 and their
existing evidence; they were not silently re-derived.

This is the living work contract for turning the local GLaDOS TTS upstream
project into a downloadable Rust command-line application with one optimized
native Torch inference path. The historical comparison work remains available
on `backend-comparison` but is not a `main` product dependency.

## Plan rules

1. Keep the command surface, model manifest, conversion evidence, and
   acceptance matrix synchronized.
2. Use [ ] pending, [~] in progress, [x] complete, and [!] blocked or awaiting
   an explicit decision.
3. Maintain one current focus.
4. Distinguish reference parity, bounded tests, empirical measurements,
   experiments, hypotheses, and non-claims.
5. Do not ship a hidden Python checkout or silently download unverified model
   files.
6. Do not claim Burn parity from a successful WAV file alone; compare
   intermediate tensors and final waveforms against the Python reference.
7. Keep backend-specific tensor ownership and model formats behind a narrow
   GLaDOS inference contract; do not force Vulkan, LibTorch, and Burn through
   one fake common tensor type.
8. Treat warm, correctness-checked benchmark evidence as the basis for
   backend selection; do not make `auto` choose from a single successful WAV.
9. Return command reports through `CliOutput` and the global `--output-format`
   renderer; do not create a doctor-specific output protocol.
10. Keep doctor diagnostic reports safe for text, JSON, and CSV rendering:
    report secret availability and provenance, never secret values, and do not
    make diagnostics mutate configuration or repair the installation.

## Intent audit evidence

- **Pass 1 — extraction:** Reread the available user instructions, the
  existing T1-T16 ledger, and the sibling `teamy-transcriber` handoff; T17-T21
  capture the explicit backend abstraction, Python-free LibTorch runtime,
  Ash/Vulkan experimentation, 4090-specific tuning, profile-based selection,
  and shared model-artifact boundary.
- **Pass 2 — traceability:** Mapped T17-T21 to G15-G19, W14-W19, and A17-A22;
  the existing Burn, model, CLI, artifact, and distribution requirements
  remain covered by T1-T16 and A1-A16. The current implementation evidence is
  attached to W17-W19 rather than promoted to a broader support claim.
- **Pass 3 — adversarial omission:** Checked that the extension does not turn
  `weavy` into an assumed GPU compiler, does not require Python for LibTorch
  inference, does not promise cross-vendor Vulkan support, does not reuse the
  sibling Whisper artifacts as TTS artifacts, and does not make performance
  evidence replace parity evidence. The Vulkan result is described as a
  Burn-acoustic hybrid until the acoustic graph is actually ported.
- **Known source limitation:** The local evidence is sufficient for the
  current backend contract and hybrid Vulkan candidate, but not for an
  upstream-class full-Vulkan acoustic implementation. The postnet and vocoder
  are now Vulkan-resident within one recorded batch; the remaining deliberate
  boundary is the Burn predictor/prenet prefix and the host-visible staging
  used only to upload batch inputs and read final outputs.
- **Burn backend limitation:** Burn WGPU and explicit Burn Vulkan compile and
  pass the waveform gate, while Burn tch compiles against the isolated
  PyTorch 2.9.0+cu128 runtime but fails the strict shape gate by one mel frame
  for one corpus item. Burn 0.19 selects `tch` 0.22, so it remains distinct
  from the upstream PyTorch 2.0.1-compatible direct TorchScript bridge. Burn
  Vulkan uses a repository Cargo patch to the matching CubeCL v0.8.1 sources
  because `cubecl-spirv` 0.8.1 is not published to crates.io; it remains
  distinct from both Burn WGPU and Ash Vulkan.
- **Doctor-slice audit:** Pass 1 extracted the new requirements as T31-T34;
  Pass 2 maps them to G20-G23, W28-W31, and A31-A33; Pass 3 checked that the
  design preserves the template's one-time `CliOutput` emission, treats
  `#[facet(sensitive)]` as a defense-in-depth annotation rather than a reason
  to serialize secrets, and keeps self-repair and model embedding out of this
  goal.

## User guidance ledger

| ID | Requirement or intent | Status | Traceability |
|---|---|---|---|
| T1 | Create a new teamy-tts repository. | Confirmed | Scope, W1 |
| T2 | Provide teamy-tts say --model glados "hello!" with explicit or automatic output paths and local playback. | Confirmed | CLI contract, W10 |
| T3 | Provide teamy-tts model list. | Confirmed | CLI contract, W9-W10 |
| T4 | Provide teamy-tts model prepare glados; the braces in the request are treated as a placeholder for a model identifier. | Confirmed, native-bundle form working; normal users use acquire-prepared | CLI contract, W9-W10 |
| T5 | Make the CLI downloadable and runnable without G:\ml\glados-tts-upstream. | Confirmed | G1, G9, W11 |
| T6 | Implement the application in Rust. | Confirmed | Architecture, W1-W10 |
| T7 | Use Burn for the neural inference path. | Confirmed direction | G3, W6-W8 |
| T8 | Convert the GLaDOS TTS upstream pipeline rather than inventing a different voice/model. | Confirmed | Reference pipeline, W3-W8 |
| T9 | Preserve local model preparation, caching, device diagnostics, and versioned assets. | Confirmed from the previous planning direction | G1, G4, W9 |
| T10 | Use teamy-rust-cli as the CLI foundation and quality-gate template. | Confirmed from the prior discussion | G8, W1, W10-W11 |
| T11 | Keep the first product a focused CLI; GUI/tray work is not part of this goal. | Proposed scope | Non-goals |
| T12 | Make the Python reference an oracle and development tool, not an undisclosed runtime dependency. | Proposed product boundary | G2, W3, W11 |
| T13 | Use the supplied G:\Programming\Repos\teamy-tts\models.zip as the raw upstream model archive. | Confirmed | Evidence, W5A |
| T14 | Rehost the raw archive in Cloudflare R2. | Confirmed direction | G12, W5B-W5C |
| T15 | Add model acquire-unprepared with Teamy and R2D2FISH-OneDrive source selectors. | Confirmed | CLI contract, W5A, W9 |
| T16 | Use Terraform to create the R2 infrastructure and ensure the archive is present and verified. | Confirmed direction | G12, W5B-W5C, A13-A15 |
| T17 | Design teamy-tts around a common framework/backend abstraction so Burn, LibTorch/TorchScript, and Vulkan can be profiled and selected rather than hard-coding one inference implementation. | Confirmed | G15, W14, W16, A17-A21 |
| T18 | Keep LibTorch/TorchScript as a known candidate because the current Rust-to-C++ bridge already runs inference without launching Python; make its native runtime dependency explicit for build and distribution. | Confirmed | G16, W15, A18-A19 |
| T19 | Explore handwritten GPU kernels through `G:\Programming\Repos\ash` and Vulkan, using `G:\Programming\Repos\facet\weavy` only where its host-side IR/JIT ideas are relevant; do not assume either project already provides a TTS GPU backend. | Confirmed direction | G17-G18, W17-W18, R16 |
| T20 | Permit aggressive specialization for the RTX 4090, including device-specific layouts, cooperative-matrix paths, and fixed-shape inference, while keeping support claims honest. | Confirmed | G17, W17-W19, A20-A22 |
| T21 | Set an explicit goal to make the multi-backend runtime, profiling/selection path, and working Burn/LibTorch/Vulkan candidates operational end to end. | Confirmed | Purpose, W14-W19, overall criteria |
| T22 | Make runtime prerequisites usable without per-invocation environment setup: expose remembered configuration through the CLI, keep environment variables as overrides, and make the Windows installer self-contained for native LibTorch DLL loading. | Confirmed | W20, A23 |
| T23 | Benchmark explicit Burn placements alongside the existing LibTorch and Vulkan candidates, with each result keyed by concrete backend identity and correctness-gated before automatic selection. | Confirmed | W21, A24 |
| T24 | Include Burn's native tch, WGPU, and explicit Vulkan backends as distinct comparison candidates; do not conflate Burn tch with the direct LibTorch bridge or Burn WGPU/Vulkan with Ash Vulkan. | Confirmed; WGPU and Burn Vulkan pass the gate, Burn tch is measured but shape-gated out | W21, A24, R22-R23 |
| T25 | Keep a representative long-form benchmark workload that records warm latency, model-load latency, output duration, real-time factor, and recurrent output frame counts without weakening the correctness-gated automatic-selection corpus. | Confirmed; `glados-long-v1` is diagnostic-only when `--skip-correctness` is used | W22, A25 |
| T26 | Optimize Burn recurrent inference by packing compatible gate projections while preserving the serialized Burn module layout and generic backend fallback. | Confirmed checkpoint; packed bidirectional GRU and LSTM pass the short waveform gate; kernel-level fusion remains open | W22, A26 |
| T27 | Add the first backend-native fused Burn recurrent kernel without changing Burnpack artifacts or weakening portable fallback behavior. | Confirmed; plain CUDA uses a specialized `CubeCL` bidirectional LSTM kernel, while fused/all-backends builds retain the packed fallback | W22, A27 |
| T28 | Freeze the final single-backend native Torch toolchain before cleanup: exact `tch`/`torch-sys`, LibTorch/PyTorch, CUDA toolkit, compiler, and runtime packaging versions; do not upgrade Burn because it is comparison-only. | Confirmed; `tch`/`torch-sys` 0.24.0 + LibTorch 2.11.0+cu128, MSVC, and the Windows CUDA-link workaround are validated on the RTX 4090 | W25, A28 |
| T29 | Remove non-tch backend implementations from `main`, migrate the prepared bundle to TorchScript artifacts, and provide one benchmark command for the remaining runtime. | Confirmed in source; the historical implementations remain on `backend-comparison` | W26, A29 |
| T30 | Rehost the tch-native bundle and rehearse the installed executable from a clean machine/PATH. | Local rehearsal complete; public rehosting and remote acquisition remain pending explicit rights and authorization | W27, W34, A30, A34 |
| T31 | Make a diagnostic `teamy-tts doctor` command the next implementation goal; it should describe system health without attempting brittle self-repair. | Implemented; report generation and checks are non-mutating | G20-G23, W28-W31, A31-A33 |
| T32 | Reuse the teamy-rust-cli `CliOutput` and top-level `--output-format` behavior so doctor is an ordinary typed command, not a special output path. | Implemented; text/JSON use the typed report and CSV uses a shared flat projection | G20, W28, A31 |
| T33 | Use Facet-derived report types and `#[facet(sensitive)]` where appropriate, but never place raw tokens, credentials, or other secret values in doctor reports. | Implemented with safe projections and text/JSON/CSV secret tests | G21, W28-W29, A31-A33 |
| T34 | Keep model weights and native runtime files external and independently updateable; doctor should inspect local model files, model-server availability, LibTorch, CUDA, and audio prerequisites. | Implemented; model/runtime checks are external and diagnostic-only | G22-G23, W29-W31, A32-A33 |

## Evidence inspected

### GLaDOS upstream

Source: G:\ml\glados-tts-upstream

Repository evidence:

- origin is https://github.com/TeamDman/glados-tts.
- upstream is https://github.com/R2D2FISH/glados-tts.
- local HEAD is commit 3447811, Switch to uv.
- the local worktree was clean when inspected.
- the upstream is MIT licensed, but model-file redistribution rights and the
  GLaDOS voice identity require a separate release decision.

Runtime evidence from the local virtual environment:

- Python torch version is 2.0.1+cu117.
- CUDA is available on the inspected machine.
- glados-new.pt is a TorchScript module with forward, generate_jit, _pad,
  and _generate_mel methods.
- the vocoder files are TorchScript modules with forward.
- glados_p1.pt and glados_p2.pt are tensors shaped [1, 256].
- prepare_text("hello!") produced six int64 token IDs.
- CPU reference inference produced mel_post shaped [1, 80, 56] and audio
  shaped [1, 1, 14336].
- the Python writer emits mono 22050 Hz int16 WAV audio.
- The warmed upstream Python path measured approximately 61 ms for the
  acoustic graph and 19 ms for the CUDA vocoder on the inspected RTX 4090.
- The warmed teamy-tts Burn path measured approximately 2.1 s for the CPU
  acoustic graph and 1.9 s for the CUDA vocoder, or approximately 4.0 s total.
- The available `tch` bindings generate against a newer C++ API than the
  installed PyTorch 2.0.1 headers. A feature-gated LibTorch C++ bridge is now
  retained for the local fast path; it loads the upstream TorchScript pair,
  keeps the model resident, and uses CPU acoustic inference plus a CUDA
  vocoder to match the upstream device split. Burn remains the portable
  product path.

Important source files:

- G:\ml\glados-tts-upstream\glados.py
- G:\ml\glados-tts-upstream\engine.py
- G:\ml\glados-tts-upstream\utils\tools.py
- G:\ml\glados-tts-upstream\utils\text\cleaners.py
- G:\ml\glados-tts-upstream\utils\text
- G:\ml\glados-tts-upstream\models

### Current model assets

| Role | File | Approximate size | Current evidence |
|---|---|---:|---|
| ForwardTacotron candidate | glados-new.pt | 116 MB | TorchScript, exposes generate_jit |
| Older Tacotron candidate | glados.pt | 76 MB | TorchScript, same public method names |
| HiFiGAN GPU | vocoder-gpu.pt | 56 MB | TorchScript |
| HiFiGAN CPU high quality | vocoder-cpu-hq.pt | 56 MB | TorchScript |
| HiFiGAN CPU low quality | vocoder-cpu-lq.pt | 5.9 MB | TorchScript |
| Speaker embedding 1 | emb/glados_p1.pt | 1.8 KB | [1, 256] float32 tensor |
| Speaker embedding 2 | emb/glados_p2.pt | 1.8 KB | [1, 256] float32 tensor |
| English phonemizer | en_us_cmudict_ipa_forward.pt | 66 MB | DeepPhonemizer checkpoint |

The first model package is therefore roughly 250 MB before compression,
depending on which vocoder variant is selected. The binary must not embed
these weights by default.

### Supplied raw archive

The file G:\Programming\Repos\teamy-tts\models.zip was inspected without
extracting it:

- size: 343,345,374 bytes;
- SHA-256:
  AFB60DD8944934EA5C67BD85DE70F424C151B5F41B50DC039578716364FA68C4;
- entries: models/emb/glados_p1.pt, models/emb/glados_p2.pt,
  models/en_us_cmudict_ipa_forward.pt, glados-new.pt, glados.pt,
  vocoder-cpu-hq.pt, vocoder-cpu-lq.pt, and vocoder-gpu.pt;
- each file size matches the corresponding upstream file inspected in
  G:\ml\glados-tts-upstream\models.

This archive is a release input and upload artifact, not a Git-tracked source
file. It is ignored by the repository and must be published only after the
model/voice redistribution decision is recorded.

### Cloudflare publication evidence

The intended Teamy source is an immutable object in a Cloudflare R2 bucket,
published through a public HTTPS download endpoint. For development, the
Terraform-managed `r2.dev` domain is the first endpoint; a custom Cloudflare
domain remains the production distribution path. The CLI must not contain R2
access keys or depend on the S3 API for normal downloads.

Cloudflare's current Terraform guidance separates responsibilities:

- the Cloudflare Terraform provider creates and manages the R2 bucket;
- the AWS-compatible provider is used for R2 S3 configuration such as
  lifecycle rules and, if suitable, object management;
- large-object publication should use an S3-compatible multipart uploader and
  a post-upload HEAD/checksum verification;
- the supplied archive is approximately 343 MB, so a single-request or
  Wrangler-only upload path is not the default plan.

References:

- https://developers.cloudflare.com/r2/examples/terraform/
- https://developers.cloudflare.com/r2/examples/terraform-aws/
- https://developers.cloudflare.com/r2/objects/upload-objects/
- https://developers.cloudflare.com/r2/api/s3/presigned-urls/

### Template and prior art

The CLI foundation is G:\Programming\Repos\teamy-rust-cli:

- figue and facet argument parsing;
- text, JSON, and CSV output through one top-level renderer;
- app-home and cache-home resolution with environment overrides;
- structured tracing and optional NDJSON logs;
- cancellation;
- Git/version/build metadata;
- Windows resources;
- CLI argument fuzzing and round-trip tests;
- check-all.ps1 quality gate.

The model conversion prior art is G:\Programming\Repos\whisper-burn:

- explicit model conversion tools;
- Burn model records and configuration files;
- model directories keyed by a user-facing model name;
- backend selection;
- separate model preparation from inference.

The prior art is a starting point, not proof that Whisper's conversion
approach can consume these GLaDOS TorchScript files.

### Cross-project model-artifact decision

The sibling `teamy-transcriber` implementation provides a compatible model
root convention: `TEAMY_TRANSCRIBER_MODEL_DIR` can override the model root,
and a prepared revision is inspected as a self-contained directory containing
an explicit manifest plus native artifacts. Its preferred Whisper layout is
`model.bpk`, `dims.json`, and `tokenizer.json`; legacy packed-NPY directories
remain discoverable for migration.

teamy-tts adopts the compatible parts: `TEAMY_TTS_MODEL_DIR`, model roots
keyed by stable model ID and revision, and a required prepared `manifest.json`
that records artifact roles, hashes, shapes/dtypes, backend support, and
converter version. It does not reuse Whisper's `dims.json` or tokenizer
contract: GLaDOS has separate ForwardTacotron and HiFiGAN networks plus a
DeepPhonemizer/IPA frontend. The TTS prepared package may therefore contain
multiple role-specific Burnpack records and frontend assets under the same
manifest-controlled revision directory.

### Current backend evidence

The current runtime now has an explicit workload-level candidate boundary:

- `src/backend.rs` defines `GladosBackend`, `SynthesisInput`, stable
  `BackendKind` identities, and `BackendSelection` policy parsing.
- `src/runtime.rs` owns a `Box<dyn GladosBackend>` and keeps tokenization,
  voice selection, and final audio ownership outside backend-specific tensor
  types.
- Burn, the existing TorchScript implementation, and the feature-gated Vulkan
  hybrid implement the same contract. The Vulkan hybrid dispatches the actual
  prepared acoustic embedding through Ash, gates it against Burn, and retains
  the rest of the Burn graph as an explicit reference fallback.
- `src/native_glados/torchscript.rs` owns a loaded upstream model pair and
  calls a small C++ LibTorch bridge; it does not invoke the Python
  interpreter.
- `build.rs` compiles that bridge only under the `torchscript` feature and
  requires an explicit `LIBTORCH` installation, so the candidate is working
  locally but is not yet a self-contained release target.
- The current Burn path keeps the acoustic model on `NdArray` CPU and uses
  CUDA for the vocoder by default; it is the native correctness/reference
  implementation, not the upstream-latency implementation.
- `src/runtime.rs` now exposes separate Burn identities for `burn-ndarray`,
  the existing `burn` CPU-acoustic/CUDA-vocoder split, `burn-cuda-acoustic`,
  and the feature-gated `burn-cuda-fused` build. The generic Burn runtime keeps
  acoustic and vocoder tensor ownership independent, so the CUDA acoustic
  candidate does not introduce a host mel round-trip.
- Diagnostic receipts on the inspected RTX 4090 measured approximately
  `40,491 ms` median for Burn NdArray, `2,080 ms` for the current hybrid, and
  `339 ms` for all-CUDA Burn over the two-text `glados-short-v1` corpus. The
  all-CUDA result passed the waveform gate with `relative_rms_error=0.059878`.
  These are one-warmup/one-measurement runs and are not release performance
  claims.
- The separate fusion/autotune build measured a steady-state diagnostic but
  failed correctness because its first output length was 41,216 samples versus
  the NdArray reference's 40,960. Its receipt is persisted as failing evidence
  and cannot influence `auto`.
- The same matrix run measured the existing native LibTorch bridge at about
  `340 ms` and the Ash/Vulkan candidate at about `834 ms`; both passed the
  shared waveform gate. Burn WGPU and Burn Vulkan are now distinct measured
  candidates, with Burn Vulkan passing the gate at about `4,759 ms` including
  first-use CubeCL/SPIR-V compilation. Burn tch is also measured at about
  `492 ms` using an isolated PyTorch 2.9.0+cu128 LibTorch runtime, but it
  reproduces the existing fused-Burn shape issue: `41,216` samples versus the
  NdArray reference's `40,960` for one corpus item. Neither Burn candidate is
  silently represented by the existing Ash or direct-LibTorch identities.
- The fused Burn duration-shape drift is fixed at the shared frame-conversion
  boundary. Duration values within `0.004` of a half-frame use PyTorch-style
  tie-to-even rounding, while values outside that numerical boundary retain
  the existing half-up behavior. The previous fused result (`161` frames,
  `41,216` samples) now matches the NdArray reference (`160` frames,
  `40,960` samples) for the affected corpus item; the current fused receipt
  passes with `relative_rms_error=0.070197`.
- The portable packed Burn recurrent path keeps the original `Gru` and `BiLstm`
  module records for Burnpack compatibility, but combines each direction's gate
  weights into one input and one hidden projection per timestep. Plain
  `--features cuda` now adds a separate backend-native `CubeCL` bidirectional
  LSTM kernel over Burn's CUDA tensor handles. It uses one launch per direction,
  shared hidden/cell state, and computes all four gate projections inside the
  recurrent kernel; CUDA builds that enable Burn fusion, WGPU, or Burn Vulkan
  intentionally retain the packed fallback because those features change the
  `Cuda` primitive alias. The model record and generic backend boundary do not
  change.
- On the inspected RTX 4090, the latest custom plain-CUDA kernel's one-warmup/
  one-measurement short receipt reports `271.804 ms` median and passes with
  `relative_rms_error=0.064531`; its latest long diagnostic reports
  `1,930.462 ms` and `real_time_factor=0.2247` for 8,591 ms of audio. The
  corresponding packed Burn receipts were `270.525 ms` short and `2,142.106 ms`
  long; repeated one-pass measurements vary with GPU/system load. Direct LibTorch
  remains the performance oracle at `74.186 ms` short and `371.905 ms` long,
  with short-corpus `relative_rms_error=0.000168`.
- The new `glados-long-v1` diagnostic contains the long sentence that exposed
  recurrent scaling and frame-count drift. It produced `189,440` samples
  (`8,591 ms`) and `740` acoustic frames. The packed Burn CUDA path measured
  `2,142.106 ms` warm (`real_time_factor=0.2493`); the latest plain-CUDA
  `CubeCL` kernel measured `1,930.462 ms` (`real_time_factor=0.2247`); direct LibTorch
  measured `371.905 ms` (`real_time_factor=0.0433`). Long-corpus waveform
  comparison is intentionally skipped because the Burn NdArray reference
  takes several minutes at this frame count; the short corpus remains the
  correctness gate. Diagnostic receipts with skipped correctness are never
  eligible for `auto`.
- `src/backend_receipts.rs` now keys benchmark evidence by prepared-model hash,
  device/GPU driver identity, backend build revision, and benchmark workload.
  The backend revision also includes a deterministic source fingerprint, so
  different dirty-worktree shader or Rust edits cannot reuse one another's
  receipt.
  `backend list` reports candidate availability, while `backend benchmark`
  writes correctness-gated receipts and `auto` ignores stale or failing ones.
- A real portable Burn benchmark run with explicit writable cache/model-root
  overrides wrote a receipt under `target/backend-receipt-test-cache`; its
  one-pass non-default configuration is intentionally not eligible for the
  default `auto` policy.
- A real Rust-process LibTorch benchmark on the RTX 4090 wrote both a one-pass
  diagnostic receipt and the default two-warmup/three-measurement receipt
  under `target/libtorch-receipt-test-cache`. The default receipt passed the
  Burn waveform gate with `relative_rms_error=0.000126` and selected LibTorch
  at `78.590 ms` median through `backend list`; the bridge launched no
  Python process. After the build metadata fix, a fresh receipt under
  `target/backend-receipts-metadata-fix` passed at `132.918 ms` median with
  the same `relative_rms_error=0.000121` gate.
- `tools/package-libtorch-runtime.ps1` now stages the release executable and
  matching LibTorch native DLLs (excluding `torch_python.dll` by default) and
  emits a hashed runtime manifest. The clean-path rehearsal has passed with
  the staged executable, 35 native DLLs, and no upstream Python checkout on
  `PATH`.

The local `ash` repository is a lightweight Vulkan API wrapper, not a tensor
runtime or kernel library. The local `facet\weavy` project provides a
host-side lowered IR and copy-and-patch JIT, not SPIR-V generation. The local
`HowToVulkan` project demonstrates runtime Slang-to-SPIR-V compilation and is
an implementation reference for Vulkan shader experiments. These facts keep
the Vulkan work scoped to a new backend rather than implying that an existing
GPU abstraction can be reused unchanged.

## Product boundary

### Purpose

teamy-tts turns a text string into a locally generated, deterministic-enough
GLaDOS-style WAV through a verified local model package.

### First release

1. Downloadable Windows CLI initialized from teamy-rust-cli.
2. Known-model catalog and installed/prepared status.
3. Named acquisition of the raw upstream archive from Teamy or the
   upstream-maintainer source.
4. Idempotent model prepare with checksums, progress, and clear failure output.
5. Local text normalization and phonemization.
6. Local inference for the GLaDOS text frontend, ForwardTacotron, and HiFiGAN
   through the backend contract, with Burn retained as the reference path.
7. Mono 22050 Hz WAV output.
8. A common inference-backend contract with Burn, LibTorch/TorchScript, and
   Vulkan candidates; each candidate is selectable and benchmarked, while
   `auto` chooses only an evidence-backed candidate available on the machine.
9. Structured diagnostics for model, device, timings, output path, source, and
   provenance.

### Non-goals

- A GUI, system tray application, or audio editor.
- A hosted TTS service.
- A generic multi-speaker TTS training platform.
- A requirement to preserve the upstream Flask server protocol.
- Silent use of the upstream checkout, its virtual environment, or a
  developer's current working directory.
- Supporting every upstream model variant in the first release.
- Claiming that all TorchScript operations can be imported automatically into
  Burn.
- Promising cross-vendor Vulkan performance; the first Vulkan target is the
  inspected RTX 4090, with other devices explicitly reported as unvalidated.

## Reference inference contract

The current Python reference is:

~~~text
input text
  -> append terminal punctuation when needed
  -> Unidecode
  -> expand numbers and selected abbreviations
  -> DeepPhonemizer en-us checkpoint
  -> filter to the upstream IPA symbol set
  -> collapse and trim whitespace
  -> map phonemes to integer IDs
  -> select speaker embedding p1 or p2
  -> ForwardTacotron generate_jit(alpha)
  -> mel_post [1, 80, time]
  -> HiFiGAN vocoder
  -> clamp/scale float waveform to int16
  -> mono 22050 Hz WAV
~~~

The Rust implementation must preserve this sequence unless a decision log
records an intentional compatibility change.

### Conversion boundary

The .pt files are TorchScript artifacts, not a Burn model record. The plan
therefore requires a development-time extraction/conversion pipeline:

1. Load each TorchScript module in the Python reference environment.
2. Extract weights, constants, tensor shapes, operator configuration, and
   model variant metadata into a deterministic interchange format.
3. Reconstruct the model in Rust/Burn with explicit layer definitions.
4. Save a Burn-native model record plus a manifest.
5. Compare intermediate tensors and output audio against the Python oracle.

The converter may use Python and PyTorch during development. The packaged
teamy-tts say command must not need Python unless G2 is explicitly reopened.

### Frontend gate

The upstream text frontend is not a trivial tokenizer. DeepPhonemizer,
Unidecode, inflect-based number normalization, abbreviation expansion, and the
IPA symbol inventory all affect the model input.

G2 must choose one of:

- port or convert the required DeepPhonemizer path to Rust/Burn;
- ship a documented local frontend sidecar with its own verified runtime;
- restrict the first CLI to an explicitly documented phoneme-input mode while
  the English frontend is being ported.

The normal user-facing say command is not complete until ordinary English
text works without an undocumented source-checkout dependency.

## CLI contract

### Required commands

~~~text
teamy-tts --help
teamy-tts --version
teamy-tts model list
teamy-tts model show glados
teamy-tts model acquire-unprepared Teamy
teamy-tts model acquire-unprepared R2D2FISH-OneDrive
teamy-tts model acquire-prepared Teamy
teamy-tts model acquire-prepared R2D2FISH-OneDrive
teamy-tts model prepare glados --source-dir <native-bundle-directory>
teamy-tts write --model glados "hello!" --output greeting.wav
teamy-tts say --model glados "hello!"
teamy-tts say --model glados --phonemes "eɪ"
teamy-tts phonemize --model glados "The letter A"
teamy-tts say --model glados "hello!" --output-dir <directory>
teamy-tts say --model glados "hello!" --output-dir <directory> --output greeting.wav
teamy-tts interactive --model glados --output-dir <directory>
~~~

### Initial synthesis options

- `write` and `say` accept `--model <id>`, default `glados`;
- `write` and `say` accept one positional text value;
- `write`, `say`, and `interactive` accept `--phonemes` to treat input as
  validated GLaDOS IPA-like symbols instead of ordinary English text;
- `phonemize` accepts ordinary text and reports the exact produced phoneme
  sequence and model token IDs without loading the acoustic model or vocoder;
- `interactive` reads newline-delimited text from stdin and loads the model
  once for the session;
- `write` and `say` accept `--output <path>`, an explicit output path, or a
  filename relative to `--output-dir`;
- `write` and `say` accept `--output-dir <directory>`; `interactive` accepts
  `--output-dir`; output persistence is opt-in and no command creates a
  default `outputs` directory;
- --voice <p1|p2>, default recorded in the model manifest;
- --alpha <positive number>, forwarded to the upstream duration predictor;
- --device <auto|cpu|cuda|wgpu>, accepted only for supported backends;
- --output-format wav, with other formats deferred.

Empty text, unsupported device/model/voice, non-positive alpha, and output
paths that cannot be created must fail before model execution.

### Model commands

model list reports the catalog, installed files, preparation state, supported
backends, approximate size, and license/provenance status.

model show reports the manifest and the exact files required.

model acquire-unprepared downloads the raw upstream archive from a named
source into a staging directory, verifies archive size and SHA-256 against the
source manifest, and records the acquisition receipt. Teamy points at the
Cloudflare R2 publication; R2D2FISH-OneDrive points at the upstream-maintainer
source.

model acquire-prepared downloads the catalogued native bundle archive from the
corresponding named source, verifies its size and SHA-256, records a separate
native-bundle acquisition receipt, and installs the six prepared artifacts
atomically. A normal user therefore runs acquire-prepared once and can then
run say without a separate prepare step.

model prepare consumes either a converter-produced native bundle directory or a
six-artifact native bundle ZIP. The Rust path extracts only the fixed root
artifact names, verifies all hashes while installing the prepared manifest,
and reports readiness for say. It remains the local/advanced path. The acquired
raw TorchScript archive remains the development converter's input.

`--output-dir` selects the directory for automatic numbering or relative
`--output` filenames. Without `--output-dir`, `--output` is the complete path.
`write` requires one of those destinations and emits the written WAV path on
stdout without playing it. `say` and `interactive` synthesize an in-memory
PCM16 WAV and play it synchronously; they emit a path only when persistence was
requested. Structured tracing logs and timings go to stderr; stdout is
deliberately path-only for these synthesis commands.

The command should be safe to rerun and should never treat a partially
downloaded directory as a prepared model.

## Architecture

~~~mermaid
flowchart LR
    C[figue/facet CLI] --> A[typed application actions]
    A --> R[model registry sources and doctor]
    A --> F[text normalization and phonemization]
    A --> B[common GLaDOS backend contract]
    B --> B1[Burn reference]
    B --> B2[LibTorch/TorchScript]
    B --> B3[Vulkan/Ash]
    B --> O[waveform validation and WAV writer]
    S[Teamy R2 or upstream source] --> R
    R --> U[raw archive staging and verification]
    U --> P[prepared model package]
    P --> F
    P --> B
    A --> L[structured logs and receipts]
    X[Python reference and converter] --> E[Burn-native records]
    E --> P
~~~

Suggested initial modules, preserving the teamy-rust-cli single-package
shape until crate boundaries prove necessary:

- src/cli: commands and shared output;
- src/paths: app home, cache home, model cache;
- src/model_registry: catalog, manifests, preparation, checksums;
- src/model_sources: Teamy/R2D2FISH-OneDrive source descriptors, resumable
  archive download, and acquisition receipts;
- src/frontend: normalization, IPA symbols, token IDs;
- src/tacotron: generated Burn model and inference;
- src/vocoder: generated Burn HiFiGAN model and inference;
- src/backend: backend contract, candidate discovery, benchmark results, and
  explicit/automatic selection;
- src/backend/burn: existing Burn reference implementation;
- src/backend/libtorch: Python-free LibTorch/TorchScript adapter;
- src/backend/vulkan: Ash device/runtime, shader pipelines, and 4090-tuned
  compute kernels;
- src/audio: waveform validation and WAV writing;
- src/reference: receipt and parity-test helpers, not shipped inference;
- tools/: Python extraction/conversion utilities and model inspections.

Every CLI command should use a directory module and a
command_command_command_cli.rs implementation file as required by the
teamy-rust-cli AGENTS.md convention.

## Storage and model package

Use the teamy-rust-cli app/cache resolution pattern with names changed for this
application:

~~~text
app-home/
  config/
  logs/
cache-home/
  raw-models/
    glados/
      <archive-sha256>/
        models.zip
        acquisition.json
  models/
    glados/
      <revision>/
        manifest.json
        frontend/
        tacotron/
        vocoder/
        embeddings/
        receipts/
  downloads/
~~~

The manifest must include model ID, revision, source URLs or source path,
SHA-256, byte count, file role, variant, expected dtype/shape, backend
support, sample rate, voice IDs, license information, and converter version.
The catalogued native bundle archive is currently 216,879,847 bytes with
SHA-256
AB663A68FB5263B8DF49F76B80812BA2692B5D1A0234A246528D65D89FD2F81F.

Model preparation must use temporary files and atomic rename, and must not
overwrite a verified revision in place.

The prepared manifest may advertise multiple backend payloads for the same
model revision. Burnpack records, upstream TorchScript files, and Vulkan
prepacked weights/shader metadata are distinct artifact roles; a backend may
be unavailable without invalidating the model revision for the other
backends. Backend selection must verify the payload hash and converter/kernel
revision before using a cached benchmark receipt.

### Cross-project artifact decision

The current `teamy-transcriber` work reinforces the same lifecycle boundary:
stable model ID, revision-keyed prepared directory, explicit manifest, per-file
hashes, and acquisition/preparation receipts. Those conventions are shared
between the projects. The artifact schemas are intentionally not shared:
transcriber Whisper uses `model.bpk` plus `dims.json` and `tokenizer.json`,
while teamy-tts uses role-specific Burnpacks, frontend data, and voice
embeddings. A Burnpack is a compatible container convention, not evidence that
the tensors or sidecars from one task can be loaded by the other. Backend
selection remains a runtime capability recorded in each project's manifest.

## Design gates

| ID | Gate | Current position | Exit evidence |
|---|---|---|---|
| G1 | Model distribution source | Upstream README points to Google Drive; a release-hosting source is preferred. | Stable source, checksums, license record, and clean-device download rehearsal. |
| G2 | Text frontend runtime | Pure Rust/Burn is the intended target; feasibility is unproven. | Ordinary English text produces the same phoneme IDs as the Python oracle, or an explicit sidecar decision is documented. |
| G3 | TorchScript-to-Burn conversion format | Use deterministic extracted tensors/config, then Burn-native records. | Converter output and schema are versioned and reloadable without Python. |
| G4 | Model variant | Start with glados-new + p2 + one vocoder variant; p1 and alternatives are capabilities. | Variant manifest, parity corpus, and default selection decision. |
| G5 | Burn backend | NdArray control, CPU-acoustic/CUDA-vocoder hybrid, all-CUDA, and a separate fusion/autotune build are explicit candidates. The hybrid and NdArray paths are the current parity controls; fusion remains rejected until its duration/output drift is explained. | Device matrix, output parity, timing evidence, and backend-selection evidence. |
| G6 | Audio contract | Mono 22050 Hz WAV int16 for the first release. | WAV header and waveform tests plus a reference output. |
| G7 | Text/length contract | Short and medium English text first; sentence splitting and long text are explicit follow-up. | Length limits, failure behavior, and corpus coverage. |
| G8 | CLI foundation | Initialize from teamy-rust-cli and preserve its quality/logging/output conventions. | Cargo build, --help/--version, output formats, clippy, tests, and check-all.ps1. |
| G9 | Weight and voice redistribution | Upstream code is MIT; model/voice redistribution is unresolved. | Written release decision and included notices. |
| G10 | Parity threshold | Numerical thresholds must be established from reference runs, not guessed. | Per-stage tensor tolerances and final waveform criteria. |
| G11 | Performance target | Do not promise realtime until measured end to end. | Cold-start, warm-start, CPU/GPU latency, memory, and output-duration report. |
| G12 | R2 infrastructure and credentials | Terraform creates the bucket; credentials remain in environment/CI secrets; Terraform apply is deployment work, not an implicit local action. | terraform fmt/validate/plan, reviewed diff, apply receipt, and secret scan. |
| G13 | Archive publication | The Teamy object has an immutable versioned key, correct content type/cache policy, expected byte count, and verified SHA-256. | Multipart upload receipt plus independent HEAD/download verification. |
| G14 | Upstream source adapter | The exact R2D2FISH-OneDrive URL and its redirect/download behavior must be recorded. | Source manifest and one successful acquisition test. |
| G15 | Backend contract | Burn, LibTorch/TorchScript, and Vulkan need one workload-level contract while retaining private tensor ownership and reusable workspaces. | Backend adapter tests and explicit backend selection without cross-backend tensor leakage. |
| G16 | LibTorch packaging | The existing bridge is Python-free at inference but requires matching native LibTorch libraries and model artifacts. | Feature-gated build/run test, dependency report, and a documented/bundled runtime policy. |
| G17 | Vulkan support boundary | Ash can dispatch Vulkan compute, but the backend must define shader compilation, device capability checks, memory ownership, and the RTX 4090 support claim. | Real compute probe, cooperative-matrix capability report where available, and honest unsupported-device behavior. |
| G18 | Vulkan model implementation | The first Vulkan path may use fixed-shape, prepacked, handwritten kernels and need not be a generic tensor framework. | Intermediate mel and final waveform parity against the oracle plus a repeatable end-to-end benchmark. |
| G19 | Backend selection policy | `auto` must choose from warm, correctness-checked evidence keyed by model/device/backend revision and must have an explicit fallback. | Benchmark receipt, selection test, and CLI diagnostics showing why a backend was chosen or rejected. |
| G20 | Diagnostic output contract | `doctor` is an ordinary command returning `CliOutput::facet(report)`; global `--output-format` remains the only output-format switch. | Text/JSON/CSV rendering works through the shared top-level emission path, including redirected stdout behavior. |
| G21 | Diagnostic data safety | Reports may expose paths, versions, hashes, presence, and redacted provenance, but never raw credentials or tokens. `#[facet(sensitive)]` is defense in depth, not the primary boundary. | Serialization tests prove text, JSON, and CSV outputs contain no secret fixtures. |
| G22 | Health-check ownership | Checks must call the same configuration, model-registry, and runtime discovery boundaries used by synthesis; shallow checks must not load heavyweight models unnecessarily. | Unit tests for pure checks, integration tests for discovery, and a real `doctor` invocation against the current installation. |
| G23 | External dependency diagnostics | Model-server availability, LibTorch/CUDA state, and audio capability are reported with bounded timeouts and offline/skip semantics; no check repairs state. | Offline and reachable/unreachable fixtures distinguish local failure, network failure, and unavailable optional capability. |

## Work breakdown

### Phase 1: CLI foundation and contract

#### W1 [x] Initialize from teamy-rust-cli

Work: Use the template initializer to create the Rust package in
G:\Programming\Repos\teamy-tts. Rename package metadata, app/cache variables,
repository URLs, Windows resources, and README. Retain figue/facet parsing,
shared output, logging, cancellation, version metadata, tests, and
check-all.ps1.

Validation: cargo run -- --help, cargo run -- --version, text/JSON/CSV output,
and the template quality gate run in the new repository.

Completion: A clean Rust CLI shell exists with no GLaDOS inference yet and no
template placeholders left in user-facing metadata.

Current state (2026-08-07): The template initializer copied the Cargo/build
metadata, Windows resources, logging, paths, output renderer, cancellation,
fuzzing, and quality-gate foundation. Package metadata now targets
`teamy-tts`, and the template's demo `init` command has been removed.

#### W2 [x] Freeze command and manifest schemas

Work: Define typed command arguments, model IDs, voice IDs, device IDs,
preparation states, output formats, error categories, manifest schema, and
structured receipt schema.

Validation: Argument fuzz/round-trip tests and JSON fixture round-trips cover
invalid combinations and unknown model revisions.

Completion: CLI parsing and model metadata are stable enough for the converter
and runtime to target.

Current state (2026-08-07): `model list`, `model show glados`,
`model acquire-unprepared`, `model prepare glados --source-dir <bundle>`, and
`say` are implemented in Rust with the template's text/JSON/CSV-compatible
top-level output. The catalog, acquisition receipt, prepared manifest,
role-specific artifact descriptors, cache-path resolution, archive
byte-count/SHA-256 metadata, voices, revision, and raw/prepared installation
states are defined.

### Phase 2: reference oracle and conversion tools

#### W3 [~] Make the Python reference deterministic

Work: Extract a small reference runner from glados.py that accepts one text,
voice, alpha, and output path; emits intermediate token IDs, phonemes, mel
metadata, waveform statistics, timings, and model fingerprints.

Validation: Reference outputs are repeatable on the inspected environment;
empty input, punctuation, numbers, abbreviations, and unsupported characters
have explicit behavior.

Current state (2026-08-07): `tools/reference_glados_frontend.py` regenerates
and checks the four-case frontend receipt in
`reference/frontend-corpus.json`. Full mel/waveform receipts and the remaining
edge-case policy are still outstanding.

Completion: The Python oracle can generate a machine-readable receipt and WAV
fixture for every parity test case.

#### W4 [~] Inventory TorchScript model contracts

Work: Record module methods, graph operations, tensor names/shapes/dtypes,
constants, layer configuration, variant differences, and operator gaps for
glados-new, glados, and each vocoder.

Validation: The inventory is generated from the files and checked into an
artifact format; every required operation has a planned Burn equivalent or a
named blocker.

Current state (2026-08-07): `tools/inspect_glados.py` generates a deterministic
JSON inventory from the upstream models directory. It records file hashes,
TorchScript methods and operator counts, acoustic state-dictionary tensor
shapes/dtypes, and the frozen-vocoder `prepacked::conv2d_clamp_run` gap. A
checked-in inventory fixture and complete operator-to-Burn mapping remain.

#### W5 [ ] Build deterministic model extraction

Work: Extract state and configuration from TorchScript or its originating
PyTorch structures into a versioned interchange directory. Include embeddings,
phonemizer data, tensor layouts, and checksums.

Validation: Extract twice and compare manifests and tensor hashes. Reload all
extracted tensors in a small Python verifier.

Completion: Conversion inputs are reproducible and independent of the
developer's current working directory.

#### W5A [~] Define raw model acquisition sources

Work: Define source descriptors for Teamy and R2D2FISH-OneDrive, including
display name, raw/native URLs, archive keys, expected byte counts, SHA-256,
content type, source provenance, redirect policy, and whether resumable range
downloads are supported.

Validation: The supplied archive is checked against the Teamy source manifest;
the upstream-maintainer URL is tested without storing credentials; an invalid
source cannot be selected silently.

Completion: The CLI can distinguish raw archive acquired, raw archive
verified, and prepared model ready.

Current state (2026-08-08): The pure-Rust CLI has raw and native source
selectors for `Teamy` and `R2D2FISH-OneDrive`, baked content-addressed Teamy
Cloudflare URLs with environment override support, catalogued size/SHA-256
contracts, content-addressed staging, cancellation checks, atomic
archive/receipt installation, and actionable unknown/unconfigured-source
errors. The native path was rehearsed by downloading the real bundle into an
empty cache, then preparing and synthesizing from its receipt. Redirect/resume
behavior and source manifests remain outstanding. The live Teamy raw and
native Cloudflare URLs have now both been downloaded and verified against
their catalogued SHA-256 values.

#### W5B [~] Add Terraform-managed Cloudflare R2 infrastructure

Work: Add an infra/cloudflare directory with pinned Cloudflare and AWS
provider versions, explicit account/bucket/location literals, an R2 bucket,
appropriate lifecycle behavior for incomplete multipart uploads, and only
outputs with a documented consumer. Keep all credentials out of the repository
and state inputs.

Validation: terraform fmt, terraform init, terraform validate, and
terraform plan run with placeholder-safe or test credentials. A review checks
that public exposure is limited to the intended model bucket and that the two
source URL outputs have a direct consumer in the Rust source catalog.

Completion: The reviewed Terraform plan creates only the intended R2
infrastructure and can be applied by an authorized operator.

Current state (2026-08-08): `infra/cloudflare` now contains pinned Cloudflare
and AWS providers, explicit deployment literals, an Azure remote state
backend, the R2 bucket, native incomplete-multipart lifecycle configuration,
content-addressed archive publication, and credential instructions.
`terraform init`, `terraform fmt-check`, and `terraform validate` pass. The
configuration now includes a Terraform-managed public development domain and
raw/native URL outputs, and the resulting URLs are baked into the Rust source
defaults with environment overrides retained. The managed domain and outputs
have been applied; live raw and native download verification is complete.

#### W5C [~] Publish and verify models.zip

Work: Add a repeatable multipart publication script or CI task using an
S3-compatible R2 client, keyed by archive SHA-256. Upload the supplied
models.zip and native bundle, attach content type/cache metadata, enable the
managed public development domain, bake its two immutable object URLs into
the Teamy source defaults, and verify each object with HEAD plus an
independent checksum/size check.

Validation: Re-run publication with the same archive, change a test archive,
interrupt a multipart upload, and verify that incomplete uploads are cleaned
up. Confirm the object can be downloaded through the exact Teamy HTTPS URL.

Completion: The Teamy source manifest points to an immutable verified object;
models.zip exists in the R2 bucket, and no R2 secret is present in the CLI or
repository.

Current state (2026-08-08): Terraform has uploaded the supplied archive and
native bundle to content-addressed R2 object keys and recorded them in remote
state. The managed-domain resource and URL outputs have been applied, and the
two concrete Teamy URLs are baked into the Rust source catalog. Independent
live raw and native downloads both verified their catalogued byte counts and
SHA-256 values; redirect/resume behavior remains outstanding.

### Phase 3: Rust/Burn model implementation

#### W6 [~] Implement the text frontend

Work: Port or convert the upstream cleaning, number expansion, abbreviation
handling, phonemization, IPA filtering, and symbol-to-ID mapping according to
G2. Keep a debug mode that prints the phoneme sequence and token IDs.

Validation: Compare every frontend stage with W3 fixtures. Include punctuation,
numbers, abbreviations, unicode/unidecode cases, empty input, and long input.

Current state (2026-08-07): the native runtime has the upstream IPA symbol
table, a prepared dictionary TSV, punctuation/word tokenization, a pure-Rust
ASCII/Latin cleaner with number and abbreviation expansion, and a Burn
forward-Transformer port for dictionary misses. The Transformer now matches
the upstream `sqrt(head_dim)` attention scaling and reproduces the checked
`hello` and `supercalifragilistic` phoneme outputs without Python. Broader
Unidecode coverage and exact whole-sentence token-ID parity remain open; the
checked Python token-ID corpus is `reference/frontend-corpus.json` and can be
regenerated with `tools/reference_glados_frontend.py`.

Completion: say can produce the exact token IDs expected by the oracle for the
supported English corpus without hidden upstream Python files.

#### W7 [x] Implement ForwardTacotron in Burn

Work: Reconstruct the inference-only generate_jit path: pitch-condition
prediction, duration prediction with alpha, pitch and energy prediction,
length regulation, LSTM, mel projection, postnet, and mel_post padding.

Current state (2026-08-07): the pure-Rust `SeriesPredictor` and
`ConditionalSeriesPredictor` building blocks, CBHG/prenet/postnet,
length-regulation, mel projection, and bidirectional LSTM path are implemented
with Burn. The full native acoustic model loads from Burnpack and matches the
Python intermediate tensors at float-level on the checked smoke input.

Validation: Compare token-derived intermediate tensors and mel_post against
the Python oracle. Test p1/p2 embeddings and glados-new before older glados.

Completion: Burn produces mel_post within the documented intermediate
tolerances for the parity corpus on the reference backend.

#### W8 [x] Implement HiFiGAN in Burn

Work: Reconstruct the selected vocoder variant, including transposed
convolutions, residual branches, leaky-ReLU, normalization/division, and final
tanh. Add a clear variant interface for CPU-LQ, CPU-HQ, and GPU-compatible
weights where supported.

Current state (2026-08-07): the selected frozen GPU HiFiGAN graph is
implemented with native transposed convolutions, residual blocks, leaky-ReLU,
and final tanh. A one-frame zero-mel comparison matches the Python waveform at
float-level within the observed CPU tolerance; broader waveform parity remains
part of W12.

Validation: Compare mel-to-waveform tensors with the Python oracle and check
waveform bounds, sample count, silence behavior, and output duration.

Completion: One documented vocoder variant generates parity-tested waveforms
from Burn mel_post without Python.

#### W9 [~] Integrate model registry and preparation

Work: Define the known catalog, source manifests, raw archive acquisition,
checksum verification, staging, atomic install, cache resolution, and device
compatibility checks. Keep raw acquisition separate from Burn preparation.

Validation: Test fresh acquisition, repeated acquisition, interrupted download,
bad archive checksum, missing file, offline cache, unknown source, fresh
prepare, and incompatible variant.

Completion: model list, model show, model acquire-unprepared Teamy, and model
prepare glados work without loading inference and produce actionable
diagnostics.

Current state (2026-08-08): `model list`, `model show glados`,
`model acquire-unprepared`, and `model acquire-prepared` are
implemented in Rust. Both
`model prepare glados --source-dir <native-bundle> --force` and
`model prepare glados --source-archive <native-bundle.zip> --force`
install and verify six role-specific native artifacts; `model prepare glados --force`
also consumes a verified cached native bundle. The archive path verifies the
catalogued bundle size and SHA-256 before extraction. `model acquire-prepared`
now performs the download, verification, and native installation in one
command. Direct conversion from the raw upstream TorchScript archive is
intentionally still a development converter step.

### Phase 4: user-facing inference

#### W10 [~] Connect synthesis commands to local Burn inference

Work: Connect CLI parsing, model preparation checks, frontend, Burn model
loading, device selection, alpha/voice options, WAV writing, output
validation, local playback, structured logs, and receipt generation.

Validation: Run the exact synthesis commands on a clean prepared cache and
compare the output against the Python reference. Test output paths, errors,
cancellation, repeated invocations, path-only stdout, stderr logging, and
interactive model reuse/playback.

Current state (2026-08-09): `write`, `say`, and `interactive` work against a
prepared native bundle, including voice/alpha selection, dictionary and
unknown-word frontend paths, Burn acoustic/vocoder loading, and mono PCM16 WAV
emission. `write` only writes; `say` writes and synchronously plays; and
`interactive` loads the model once, then repeats synthesis/playback for each
non-empty stdin line. Default output numbering, `--output-dir`, explicit output
paths, path-only stdout, and staged progress logs on stderr are implemented.
The default build now uses the Burn CUDA backend for HiFiGAN while keeping ForwardTacotron on the CPU reference backend.
The optimized release smoke command produced a 22050 Hz mono WAV; measured
warm synthesis for "Hello, friend" is approximately 4.0 s on the RTX 4090,
down from approximately 44 s on the all-CPU path. The remaining gap to the
upstream approximately 80 ms warm path is attributed to Burn CubeCL kernel
overhead versus PyTorch/cuDNN and remains a performance gate.

Completion: `write` and `say` can synthesize without the upstream checkout or
Python runtime from a distributed native bundle, and `interactive` can reuse
the loaded runtime for a line-oriented session.

#### W11 [~] Package and rehearse downloadability

Work: Build release artifacts, include notices and model preparation docs,
record version/Git metadata, and document app/cache overrides, device
selection, model sources, and troubleshooting.

Validation: Rehearse on a clean Windows environment with no upstream checkout,
no pre-existing cache, and no hidden PATH dependency. Run check-all.ps1 and
the acceptance matrix.

Completion: A new user can download the executable, run model list,
acquire-prepared Teamy, run write or say, and locate a valid WAV.

Current state (2026-08-08): `tools/package-native-bundle.ps1` produces a
six-artifact native bundle archive. Extracting that archive into an empty
model home, running `model prepare`, and running the exact `say` smoke command
produced a valid mono 22050 Hz 14336-sample WAV. The native bundle is now
publicly hosted with a baked content-addressed URL; a clean-machine executable
rehearsal and release notices remain.

### Phase 5: backend abstraction and candidate runtimes

#### W14 [x] Define the common backend contract and model-facing inputs

Work completed: Extracted the current Burn/TorchScript choice behind a
workload-level backend interface. `SynthesisInput` is the stable prepared
input; final audio remains the backend-independent output, while backend tensor
types stay private so Vulkan can retain GPU-resident buffers and LibTorch can
retain native tensors. Added explicit `burn`, `libtorch`, `vulkan`, and `auto`
selection semantics without changing the existing stdout/stderr contract.

Validation:

```pwsh
cargo test --offline --no-default-features --lib
cargo clippy --offline --no-default-features --all-targets -- -D warnings
cargo check --offline --features torchscript --lib
cargo clippy --offline --features torchscript --all-targets -- -D warnings
```

Completion criteria: Burn and the existing TorchScript implementation satisfy
the same backend contract; explicit selection is deterministic; unsupported
features fail with an actionable diagnostic; and no CLI command unloads a
backend during an interactive session. Evidence: 22 library tests pass;
default and TorchScript feature checks and clippy pass; `say --help` exposes
`--backend`; and an invalid backend fails before model loading with the stable
expected-values diagnostic.

#### W15 [x] Harden and package the Python-free LibTorch candidate

Work completed: The current C++ bridge is now behind the backend contract,
preserves loaded model reuse and warmup, and has explicit model-directory and
feature diagnostics. The supported distribution decision is an optional
Windows accelerator package: the executable is staged beside the matching
LibTorch native DLLs, while the verified model bundle remains separate. The
inference process does not start Python.

Validation:

```pwsh
$env:LIBTORCH='G:\ml\glados-tts-upstream\.venv\Lib\site-packages\torch'
$env:Path="$env:LIBTORCH\lib;$env:Path"
cargo check --offline --features torchscript --lib
cargo clippy --offline --features torchscript --lib -- -D warnings
cargo run --release --features torchscript -- write --model glados "Hello, friend"
```

Completion criteria: a clean Rust process can load the upstream TorchScript
pair, synthesize repeatedly without Python, report warm timings, pass the
parity corpus, and have an explicit reproducible native-library distribution
story.

Implementation evidence: repeated Rust-process LibTorch synthesis and the
default benchmark receipt pass on the inspected RTX 4090; the benchmark
correctness comparison reports relative_rms_error=0.000126. The staged release
rehearsal ran the packaged executable with 35 native DLLs, copied model
artifacts, and a PATH containing no upstream Python checkout. It produced a
valid WAV and an interactive two-line run reused the loaded model with
75 ms and 198 ms warm synthesis timings. The packaging script and hashed
runtime manifest establish the reproducible native-library layout.

#### W16 [x] Add backend discovery, benchmark receipts, and automatic selection

Work: Add a backend/benchmark command surface that measures cold load, warm
synthesis, stage timings, output duration, memory where available, device,
model revision, backend build identity, and correctness status. Persist a
selection receipt keyed by model hash, GPU/driver, backend revision, and
benchmark configuration. Make `auto` use only a passing receipt and fall back
to the documented reference backend.

Validation: Unit-test receipt keys and ordering; run the real Burn and LibTorch
benchmarks on the RTX 4090; compare the same utterance corpus and warmup
policy; verify explicit selection and `auto` diagnostics through the CLI.

Completion criteria: `backend list`, `backend benchmark`, and `--backend auto`
work from a clean cache, never select a numerically failing candidate, and
make the selected backend and evidence visible on stderr.

Implementation evidence: the new backend command surface, content-addressed
receipt schema, model/device/backend-revision matching, and Burn fallback are
implemented. Twenty-eight library tests pass, the empty-cache list command
works, and real Burn and LibTorch benchmarks were run with the same default
two-warmup/three-measurement corpus on the RTX 4090. The LibTorch receipt
passed the correctness gate and backend list selected it automatically at
78.590 ms median; malformed, stale, failing, and unavailable receipts remain
ineligible.

#### W17 [x] Build the Ash/Vulkan compute substrate and 4090 capability probe

Work: Add an optional `vulkan` feature and a Vulkan backend crate/module using
the local `ash` bindings. Implement instance/device/queue selection, storage
buffers, staging, descriptor/pipeline setup, synchronization, timestamp
queries, pipeline caching, and shader loading/compilation. Start with a
vector-add and matrix-multiply probe, then query cooperative-matrix support
and device limits. Use Slang-to-SPIR-V or an explicitly documented shader
compiler path; do not treat `weavy`'s host JIT as a GPU compiler.

Validation:

```pwsh
cargo check --offline --features vulkan --lib
cargo run --release --features vulkan -- backend probe
```

The real probe must execute on the RTX 4090, report supported matrix shapes,
and compare its result against a CPU reference. Unsupported Vulkan devices
must report capability failure rather than silently using an unvalidated path.

Completion criteria: the optional Vulkan backend can allocate, dispatch, time,
and validate a real compute kernel without affecting default Burn builds.

Implementation evidence: the optional Ash path compiles with clippy cleanly and
the real RTX 4090 probe reports Vulkan 1.3, compute queue family 0,
VK_KHR_cooperative_matrix with 15 supported shape/type combinations, a passing
1024-element vector-add dispatch with zero maximum absolute error, and a
passing fixed 16x16 matrix multiply with zero maximum absolute error. The
latest host CPU-to-fence timings were about 0.33 ms for vector-add and
0.11 ms for matrix multiply; the matrix dispatch reported about 2.7
microseconds from GPU timestamps with 64 valid timestamp bits. The GLaDOS graph
remains open. A persistent Ash context now caches the embedding shader,
descriptor layout, pipeline layout, and pipeline; the synthetic embedding
fixture passes with zero error. After moving the fixture to the persistent
model-buffer path, the latest RTX 4090 probe reports about 1.53 ms cold and
0.48 ms warm, including host buffer setup and synchronization. A real prepared
model invocation also passed through the explicit Vulkan runtime; the actual acoustic
embedding was dispatched through Ash and checked against Burn before the
remaining Burn reference graph produced a valid WAV.

A zero-warmup diagnostic receipt for the hybrid path also passed the
correctness gate with zero error, but measured about 49.5 seconds median over
the two-text corpus. It is intentionally stored under a test cache with a
non-default workload key and is not eligible for automatic selection.

The reusable model-oriented substrate now also has generic Conv1d and
ConvTranspose1d dispatches plus persistent model-buffer handles. Synthetic
fixtures for both kernels pass on the RTX 4090 with zero maximum error; these
are building blocks for the acoustic port, not evidence that the recurrent
acoustic graph has moved off Burn.

#### W18 [x] Implement the specialized Vulkan GLaDOS inference path

Work: Keep the first implementation fixed to the selected GLaDOS variant,
batch-one shapes, and the RTX 4090. Prepack weights, implement device-side
length regulation, cooperative-matrix/GEMM building blocks, fused four-gate
LSTM/state updates, required convolutions/postnet, and the vocoder path. Keep
intermediate tensors on the device where it improves measured latency. Add
fallbacks for unsupported optional features and preserve model-manifest hashes.

Validation: Compare token-derived intermediates, mel tensors, waveform
statistics, and final audio against W3 receipts. Run warm and cold benchmarks
beside Burn and LibTorch, including short and medium text. Use GPU timestamps
to separate dispatch, synchronization, transfer, and host overhead.

Implementation evidence so far: the explicit Vulkan backend owns a persistent
Ash context, dispatches the prepared acoustic embedding, and compares the
returned values against Burn with a 1e-4 maximum-absolute-error gate. It now
extracts the prepared Burnpack's fixed GLaDOS HiFi-GAN weights and executes
the complete four-stage Vulkan vocoder (ConvTranspose1d, residual Conv1d
pairs, activation, and final Conv1d) through Ash. The residual-block
accumulation and a final 32-channel activation-buffer sizing error were
corrected against the Burn graph before accepting the candidate.

The vocoder now records one device-resident batch: model weights persist in
the Ash context, intermediate tensors remain in batch buffers, elementwise
activation/add/scale kernels are ordered by explicit compute barriers, and
one fence is used for the final readback. With `--warmup 0 --measurements 1`,
the two-text corpus passed with `relative_rms_error=0.000062`,
`max_abs_error=0.000363`, and a `2592.547 ms` candidate median. The default
two-warmup/three-measurement receipt on the inspected RTX 4090 passed with
`relative_rms_error=0.000059`, `max_abs_error=0.000278`, and a
`2992.580 ms` median. Stage diagnostics with
`TEAMY_TTS_VULKAN_PARITY=1 --debug` stayed below `2.2e-5` relative RMS at the
final waveform trace. This remains a correctness-backed hybrid milestone, not
an upstream-class performance claim: the predictor/prenet remain Burn-backed,
and this original measurement predates the later device-local batch/model
storage work.

A first acoustic migration slice now owns the fixed-shape `post_proj` linear
stage in Vulkan. Burn's postnet tensor is transposed from `[1, frames, 512]`
to the shader's channel-major layout, the persistent Burnpack weights are
transposed into `[80, 512]`, and the returned `[80, frames]` mel passes the
real model parity gate at `max_abs_error=0.0000081` and
`relative_rms_error=0.000000336`. The post-projection dispatch remains a
synchronous host-visible migration boundary because the preceding recurrent
postnet graph is still Burn.

A second slice now dispatches the fixed-shape pitch and energy condition
projections through the persistent Vulkan Conv1d pipeline. On the real 4090
`Hi`, pitch projection parity passed at
`max_abs_error=0.000000060` / `relative_rms_error=0.000000048`, and energy
projection parity passed at `max_abs_error=0.000000477` /
`relative_rms_error=0.000000057`. The length-regulation shader also passed
the same run at zero error. These projections are currently parity-gated
migration boundaries; normal execution now assembles the conditioning tensor
from the Burn base prefix and the Vulkan projection candidates, while parity
mode compares the resulting continuation against the full Burn graph.

A third slice now covers the fixed-shape acoustic LSTM and mel boundary.
The Ash LSTM shader runs the forward and reverse four-gate recurrences in one
workgroup with persistent model buffers. On the real 4090 `Hi`, forward/reverse
interleaving passed against Burn at `max_abs_error=0.000017107` /
`relative_rms_error=0.000000985`; the Vulkan mel projection then passed at
`max_abs_error=0.000016212` / `relative_rms_error=0.000000586`. The candidate
mel is now the input to a Vulkan CBHG postnet batch. That batch owns the eight
Conv1d/BatchNorm bank branches, max-pool, both projections, four highway layers,
both reset-after GRU directions, and the post projection. The real 4090 parity
run passed the final post projection at `max_abs_error=0.000022411` /
`relative_rms_error=0.000000958`; parity mode still executes the complete Burn
acoustic graph as an oracle, while normal Vulkan execution no longer performs
duplicate recurrent or postnet work. The postnet and vocoder are now recorded
in one Vulkan batch, so post-projected mel does not round-trip through the host
before vocoder dispatch. At this point the batch still used host-visible
coherent temporary allocations; the later allocator slice moves the model and
batch working sets to device-local storage with explicit transfer staging.

The optimized release path passed the shared external Burn comparison on the
two-text corpus with `correctness_passed=true`,
`relative_rms_error=0.000187`, `max_abs_error=0.000560`, and a
one-warmup/three-measurement median of `1314.138 ms` after introducing an arena
suballocator for batch tensors. This improves the previous stabilized
merged-batch median of `1354.546 ms` by about 3%; the postnet and vocoder still
share one submission and the intermediate mel still does not round-trip through
the host. The result remains far slower than the
`118.632 ms` LibTorch receipt. The remaining performance work is profiling
  dispatch, synchronization, transfer staging, and the Burn predictor/prenet
  prefix rather than another correctness migration boundary.

The opt-in `TEAMY_TTS_VULKAN_PROFILE=1` path now reports host command-recording
time, GPU timestamp time, and arena usage for each Vulkan batch. A live `Hi`
run recorded the merged postnet/vocoder batch in `94.563 ms` of host time and
`76.187 ms` of GPU time, using 274 temporary tensor slices across 17 arenas
(`285212672` bytes reserved). This validates the measurement boundary and
shows that arena suballocation is useful but not sufficient: descriptor/kernel
dispatch overhead and the Burn predictor/prenet prefix remain material.

The next residency slice moves persistent model buffers and batch arenas to
device-local memory. Batch inputs use transient host-visible transfer sources,
and requested outputs use transient host-visible readback buffers; no
intermediate tensor is mapped by the host. A one-warmup/three-measurement
RTX-4090 receipt passed the correctness gate at `1303.979 ms` median. This is a
promising but not yet repeatable latency claim; the profile still shows kernel
and barrier overhead as the dominant remaining Vulkan work.

The next barrier slice narrows the fixed graph's inter-dispatch dependency from
shader-write to shader-read/write into shader-write to shader-read. Every
dispatch in this graph writes a distinct arena slice, so no write-after-write
dependency is required. A default two-warmup/three-measurement receipt on the
same RTX 4090 passed with `1047.4319 ms` median,
`relative_rms_error=0.000187`, and `max_abs_error=0.000560`. This is a
repeatable improvement over the earlier `1303.979 ms` device-local result, but
the Vulkan candidate remains slower than LibTorch and stays an explicit
experimental override.

The first residual-block fusion experiment was deliberately rejected. A
single shader that recomputed the first convolution for each second-convolution
tap passed the short parity trace but took about `6.1 s` of GPU time for `Hi`,
demonstrating the cost of arithmetic recomputation. A two-pass Conv1d-plus-
residual-add variant passed the short parity trace but failed to complete the
standard long benchmark within two minutes, so both variants were reverted.
This is recorded as a non-claim; the next optimization slice must preserve
intermediate reuse while reducing barriers and allocation/descriptor overhead.

Completion criteria: Vulkan produces accepted audio for the supported GLaDOS
variant, passes the parity corpus within documented tolerances, and has a
repeatable benchmark result on the target 4090. The current partial completion
boundary is the Vulkan embedding/condition-projection/LSTM/mel/postnet/post-
projection/vocoder path plus the Burn predictor/prenet prefix; parity mode
retains the full Burn oracle. W18 is complete for this explicitly documented
hybrid support boundary: the current performance decision is to keep Vulkan as
an explicit RTX-4090 candidate and let receipt-backed `auto` choose it only
when it is the fastest available passing backend. Upstream-class full-Vulkan
latency remains follow-up work, not an unbounded support claim.

#### W19 [x] Select and document the best backend for each supported machine

Work: Compare Burn, LibTorch, and Vulkan using the same correctness gate and
benchmark protocol. Choose the default `auto` policy, record where a backend
is faster or more robust, document optional native dependencies and support
limits, and preserve explicit overrides for investigation.

Validation: Re-run the benchmark receipt after a clean build and model
reinstall; verify that a stale receipt is rejected when the model, GPU, driver,
or backend revision changes; run the selected backend through `write`, `say`,
and `interactive`.

Completion criteria: the application can explain and reproduce its backend
choice, and no release claim calls the slowest or least-supported candidate
the default without evidence.

Completion notes: On the inspected RTX 4090, the latest combined release
build's default benchmark receipt passed for LibTorch at `83.0876 ms` median
with `relative_rms_error=0.000121`; the latest Vulkan receipt passed at
`903.9575 ms` median with `relative_rms_error=0.000187`. `backend list` selected LibTorch and reported
the matching revision-keyed receipt, while an automatic `write` synthesized
through LibTorch and emitted only the requested WAV path on stdout. The
release packager staged the executable and 35 native LibTorch DLLs without
`torch_python.dll`, and a clean-path packaged run synthesized a valid WAV.
The README records the support boundary: Burn is the portable default,
LibTorch is the measured Python-free CUDA candidate, and Vulkan is an explicit
experimental RTX-4090 override; its embedding, condition projections, LSTM/mel,
postnet, post projection, and vocoder are parity-gated, but its latency remains
open.

The optional Vulkan executable now has a reproducible stager at
`tools/package-vulkan-runtime.ps1`. It copies the feature-specific executable
and writes a SHA-256 manifest, while explicitly leaving the Vulkan-capable
graphics driver and verified model bundle as external prerequisites.

The durable runtime configuration boundary is now explicit:

- `src/config.rs` stores backend policy, prepared-model root, TorchScript model
  directory, and LibTorch device index under the application home in
  `config.json`.
- `teamy-tts config show|set|clear` is the user-facing configuration surface.
  Explicit command flags win over environment overrides, which win over the
  remembered values, which win over built-in defaults.
- `TEAMY_TTS_BACKEND`, `TEAMY_TTS_MODEL_DIR`,
  `TEAMY_TTS_TORCH_MODEL_DIR`, and `TEAMY_TTS_TORCH_DEVICE` are temporary
  overrides rather than required setup for ordinary installed use.
- `update.ps1` builds the tch runtime with `LIBTORCH` scoped to that build and
  copies the matching native DLLs beside the installed executable. It leaves
  remembered configuration untouched; model setup remains an explicit CLI
  operation.

#### W20 [x] Make installed runtime prerequisites durable and overridable

Work completed: Added a durable `config.json` managed through
`teamy-tts config show`, `config set`, and `config clear`. Backend policy,
prepared-model root, TorchScript model directory, and LibTorch device index
now have CLI-set remembered values. Explicit command values take priority;
environment variables remain useful temporary overrides. The runtime,
prepared-model path resolver, TorchScript device selection, and benchmark
availability checks all use the same precedence rules. Updated `update.ps1`
to set `LIBTORCH` only in its build process and copy the matching LibTorch DLLs
beside the installed executable without overwriting remembered settings.

Validation:

```powershell
cargo test --offline --no-default-features --all-targets
$env:LIBTORCH='G:\ml\glados-tts-upstream\.venv\Lib\site-packages\torch'
$env:Path="$env:LIBTORCH\lib;$env:Path"
cargo clippy --offline --no-default-features --features all-backends --all-targets -- -D warnings
```

Completion criteria: an installed all-backends executable can locate its
LibTorch DLLs beside itself, a user can configure the model directory once
through the CLI, and an environment override changes the effective value
without modifying the remembered file. The CLI tests, config smoke test, and
PowerShell parser check pass.

#### W21 [x] Benchmark the explicit Burn placement matrix

Work completed: Generalized the Burn runtime over independent acoustic and
vocoder backend types and added stable identities for `burn-ndarray`, `burn`,
`burn-cuda-acoustic`, the separate `burn-cuda-fused` Cargo build, Burn's
`burn-tch` backend, Burn's `burn-wgpu` backend, and the explicit `burn-vulkan`
candidate. Extended
backend listing, configuration/help text, availability checks, and receipt
keys so candidate results remain distinct. Correctness comparison now uses the
explicit NdArray reference, which is stable even when Burn's `Cuda` alias is
changed by the fusion feature.

Validation on the RTX 4090 (`GPU-eb5e19ad-368f-0e7d-605c-610f2f08a114`, driver
610.88) used the `glados-short-v1` two-text corpus and a writable repository
cache. Burn NdArray measured `40,491.343 ms` median, the current hybrid
`2,079.634 ms`, all-CUDA `338.820 ms`, Burn WGPU `4,266.578 ms`, Burn Vulkan
`4,758.706 ms`, Burn tch `491.884 ms`, native LibTorch `339.578 ms`, and Ash
Vulkan `833.761 ms`. Burn WGPU and Burn Vulkan passed the waveform gate. The
historical pre-W22 fusion receipt failed because the candidate/reference
sample counts differed (`41,216`/`40,960`); W22's shared duration-boundary fix
refreshed the fused short receipt to a passing `relative_rms_error=0.070197`
result. Burn tch remains explicitly failing because its
candidate/reference sample counts differ. Burn tch was
made runnable by provisioning an isolated PyTorch 2.9.0+cu128 environment;
the upstream PyTorch 2.0.1 environment remains reserved for the direct
TorchScript bridge. Burn Vulkan was made runnable by pinning the matching
CubeCL v0.8.1 sources because `cubecl-spirv` 0.8.1 is not published to the
local crates.io registry. The Ash Vulkan row remains a separate implementation.

The Burn Vulkan executable receives a 64 MiB Windows main-thread stack because
CubeCL's first SPIR-V graph compilation exceeds the default stack; this fixes a
process-level stack overflow without changing the benchmark's steady-state
work. Burn tch uses the repository-local `target\teamy-tts-pytorch-29`
environment for PyTorch 2.9.0+cu128 and is intentionally not part of
`all-backends`, because the ordinary distribution build still targets the
upstream PyTorch 2.0.1-compatible direct TorchScript bridge.

Completion criteria: each runnable candidate has a receipt or an explicit
failure/unavailable record, automatic selection ignores failing receipts, and
the plan records the corpus, device, build variant, correctness status, and
non-claim status. Portable/default/fused feature checks passed before the
measurements.

#### W22 [x] Measure and optimize Burn recurrent scaling

Work: Add a stable long-form diagnostic corpus, output-duration and acoustic
frame diagnostics, model-load/real-time-factor reporting, and an explicit
correctness-skip mode for workloads whose CPU reference is too slow for routine
comparison. Pack compatible GRU and LSTM gate projections without changing the
serialized Burn module layout. Stabilize the duration-to-frame boundary for
small backend floating-point differences, then compare the result with the
direct LibTorch oracle.

Validation: On the inspected RTX 4090, the packed Burn path must preserve the
short waveform gate, preserve output frame counts for the previously failing
fused candidate, and report a reproducible long-form measurement. Skipped long
comparisons must remain ineligible for `auto`; the completed next slice is a
backend-native fused recurrent kernel rather than additional generic tensor
launches.

Current evidence: `glados-long-v1` produces 8,591 ms of audio and 740 acoustic
frames. The packed `burn-cuda-acoustic` path measured 2,142.106 ms warm versus
LibTorch at 371.905 ms. The first backend-native CUDA `CubeCL` LSTM kernel now
measures 1,930.462 ms on the same long workload and 271.804 ms on the short
workload, passing the short waveform gate with `relative_rms_error=0.064531`.
The fused duration-shape issue is fixed by a 0.004-frame tie window with
PyTorch-style tie-to-even rounding. The remaining Burn-versus-LibTorch gap is
an explicit follow-up optimization target, not a hidden automatic-selection
claim.

#### W23 [x] Close the remaining Burn recurrent performance gap

Work: Use the validated plain-CUDA `CubeCL` LSTM kernel as the baseline for
profiling memory traffic, launch geometry, and the still-packed GRU/predictor
stages. Compare intermediate recurrent tensors against the NdArray reference
before adding more specialization. Keep the packed Burn implementation as the
portable fallback and keep LibTorch eligible as the local performance choice
until a correctness-passing Burn receipt wins on the target machine.

Validation: New kernels must preserve the Burnpack artifact layout, pass the
short waveform gate, report long-form frame counts, and show warm latency
separately from first-use CubeCL compilation.

Result: Burn-tch now specializes the recurrent and CBHG boundaries with
LibTorch tensor operations while retaining Burn-owned modules and Burnpack
artifacts. The long corpus produces 740 frames and 8,591 ms of audio; a
two-warmup/three-measurement receipt on the RTX 4090 reports `271.760 ms`
median, `316.495 ms` p95, and `real_time_factor=0.0333`. The short correctness
receipt passes with `relative_rms_error=0.071774` and `max_abs_error=0.255078`.
The portable Burn CUDA/CubeCL path remains a separate candidate and the direct
LibTorch/TorchScript path remains the oracle.

#### W24 [ ] Stabilize the Burn-tch latency tail

Work: Replace the current per-call construction of LibTorch GRU/LSTM parameter
lists with persistent, contiguous cuDNN-compatible weights owned by the loaded
Burn-tch runtime. Remove the repeated `RNN module weights are not part of
single contiguous chunk` warnings, then profile the remaining predictor, CBHG,
and vocoder launches.

Validation: Preserve the W23 short-corpus correctness receipt, repeat the long
corpus with at least two warmups and five measurements, and bring the p95
warm-latency result below 300 ms without changing the Burnpack artifact format
or making the direct TorchScript bridge a hidden dependency.

### Phase 6: evidence and follow-up

#### W12 [ ] Establish parity, performance, and release evidence

Work: Maintain a corpus and acceptance matrix covering frontend token IDs,
mel_post, waveform statistics, perceptual/audio comparisons, cold/warm
latency, memory, backend, model revision, and device. Record the upstream
TorchScript/cuDNN baseline separately from the Burn CPU and CUDA-hybrid paths,
the optional LibTorch bridge, and the Vulkan candidate. The bridge currently
measures about 74-221 ms warm for the exercised utterances; its runtime
distribution policy is now documented as an optional staged accelerator
package, while the portable Burn path remains the default clean-device claim.

Validation: Reports identify evidence type, tool versions, fixture scope,
thresholds, failures, and non-claims. Do not label sampled parity exhaustive.

Completion: Release claims are traceable to receipts and measurements.

#### W13 [ ] Add deferred capabilities only after the first release

Work: Consider additional output formats, asynchronous/streaming playback, p1
as a first-class voice, CPU-HQ/GPU variants, sentence batching, and a local
service. Line-oriented interactive synthesis and basic synchronous Windows
playback are now in the first-release slice.

Validation: Each capability gets a separate manifest/acceptance row and does
not weaken the first release contract.

Completion: Deferred work is either implemented with evidence or deliberately
removed from the roadmap.

## Acceptance matrix

| ID | Criterion | Evidence required |
|---|---|---|
| A1 | CLI is a real Rust binary initialized from the template. | Cargo metadata, --help/--version, quality gate, no template placeholders. |
| A2 | model list reports known and installed model state. | Empty-cache and prepared-cache JSON/text fixtures. |
| A3 | model prepare glados is idempotent and verifies assets. | Fresh, repeat, interrupted, and checksum-failure receipts. |
| A4 | write, say, and interactive accept the synthesis command shapes and stdout/stderr contract. | CLI round-trip tests, a write end-to-end invocation, and an interactive model-reuse/playback check. |
| A5 | ordinary English frontend matches the reference. | Token/phoneme parity corpus. |
| A6 | Burn ForwardTacotron matches the reference. | Intermediate tensor comparisons and model fingerprints. |
| A7 | Burn vocoder matches the reference. | Mel-to-waveform comparison and waveform statistics. |
| A8 | output.wav is valid mono 22050 Hz WAV. | Header, sample count, bounds, and non-silent fixture checks. |
| A9 | packaged execution needs no upstream checkout or hidden Python. | Clean-device rehearsal and dependency audit. |
| A10 | device behavior is explicit. | CPU baseline and accelerated backend report, including fallback/error states. |
| A11 | release provenance is honest. | Model URLs, SHA-256, licenses, converter version, and voice redistribution decision. |
| A12 | performance claims are bounded. | Cold/warm timings, memory, audio duration, device, and model variant. |
| A13 | Teamy raw acquisition is verifiable. | Source manifest, archive byte count, SHA-256, resumable download, and independent verification. |
| A14 | Terraform creates the intended R2 infrastructure. | Format, validate, reviewed plan, apply receipt, and secret scan. |
| A15 | models.zip exists in the Teamy bucket. | Object HEAD metadata plus independent HTTPS download/checksum verification. |
| A16 | The upstream-maintainer source remains available as a distinct option. | R2D2FISH-OneDrive source descriptor and acquisition fixture. |
| A17 | Burn, LibTorch/TorchScript, and Vulkan implement one workload-level backend contract. | Adapter tests, explicit selection, and no public dependency on backend tensor types. |
| A18 | LibTorch/TorchScript inference runs without launching Python. | Rust process trace/diagnostics, native-library dependency report, and repeated warm synthesis. |
| A19 | The selected LibTorch native runtime is reproducible for supported builds. | Clean build/run rehearsal or an explicit optional-accelerator support boundary. |
| A20 | Vulkan can execute and validate a real compute kernel on the RTX 4090. | Ash backend probe, GPU timestamps, cooperative-matrix/device capability report, and CPU comparison. |
| A21 | Vulkan GLaDOS inference is end-to-end and numerically/audio validated. | Intermediate tensor and waveform parity corpus plus warm/cold benchmark. |
| A22 | Backend selection is evidence-backed and reproducible. | Benchmark receipt keyed by model/device/backend revision, explicit override, auto-selection, and fallback tests. |
| A23 | Installed runtime configuration does not require per-invocation environment setup. | `config` CLI round trip, environment-overrides-config smoke test, all-backends install script, and LibTorch DLL co-location. |
| A24 | Burn candidate placement is measurable and cannot silently bypass parity. | Distinct receipts or explicit toolchain records for NdArray, hybrid, all-CUDA, fused, tch, WGPU, and explicit Vulkan builds; explicit NdArray reference; matrix timing and failure record. |
| A25 | Long-form recurrent scaling is measured without weakening automatic selection. | `glados-long-v1` receipt with frame count, output duration, real-time factor, model-load time, and explicit skipped-correctness status. |
| A26 | The first packed Burn recurrent optimization preserves model compatibility and correctness. | Burnpack load, duration-boundary regression test, short waveform gate, and refreshed fused receipt. |
| A27 | A backend-native fused Burn recurrent kernel preserves artifacts, parity, and warm-measurement boundaries. | Plain-CUDA `CubeCL` LSTM launch, short waveform gate, long-form frame/RTF receipt, portable fallback, and clippy/tests. |
| A28 | The selected native Torch/CUDA toolchain is pinned and reproducible. | `DEPENDENCIES.md`, clean build/run evidence, CUDA-link probe, and matching LibTorch packaging record. |
| A29 | `main` contains one direct tch/LibTorch runtime while comparison history remains preserved separately. | Source audit, all-targets build/test, model load, correctness receipt, and benchmark command. |
| A30 | The external model bundle and adjacent native runtime work from a clean cache/PATH. | Published-object verification, clean acquisition, installed-process receipt, and no upstream/Python dependency. |
| A31 | `doctor` uses the shared `CliOutput` and global `--output-format` contract. | Text/JSON/CSV CLI tests, redirected JSON parse, and no doctor-local renderer. |
| A32 | `doctor` diagnoses model, configuration, LibTorch, CUDA, audio, and model-server state with shallow/deep and offline behavior. | Focused status aggregation and source-probe tests, current-machine shallow/deep run, bounded network failures, explicit offline/unconfigured skips, and remediation evidence. |
| A33 | Doctor output is safe and non-mutating. | Text/JSON/CSV secret-fixture scan, no configuration/cache mutation, and README-documented report status/exit behavior. |
| A34 | The local installed executable and external artifacts are auditable without hidden development dependencies. | W34 receipt with archive/runtime hashes, cleared child environment, typed doctor output, cold/warm behavior, opt-in WAV write, in-memory playback, resident interactive behavior, and honest missing/corrupt model/runtime failures. |
| A35 | Playback can be tested end to end without audible output. | `--volume` validation, zero-scaling unit coverage, and W34 staged `say`/`interactive` runs that complete synchronous in-memory playback at volume zero. |

## Risks and stop conditions

| ID | Risk | Mitigation / stop condition |
|---|---|---|
| R1 | TorchScript contains prepacked or fused operations that do not map cleanly to Burn. | Stop automatic import; extract original weights/config and implement explicit equivalent layers. |
| R2 | DeepPhonemizer is harder to port than the neural synthesizer. | Keep G2 explicit; do not call the product complete with a hidden Python frontend. |
| R3 | Burn backend numerical differences alter voice quality. | Establish stage-level tolerances and keep CPU reference as the acceptance baseline. |
| R4 | Upstream model variants are not equivalent. | Treat glados-new, glados, CPU-LQ, CPU-HQ, and GPU as separate manifest variants. |
| R5 | Model files are large or unavailable from a redistributable source. | Separate binary from model package; block release until source/license/provenance is documented. |
| R6 | GLaDOS voice rights differ from code license. | Record a distribution decision; do not infer model/voice rights from MIT code alone. |
| R7 | Frontend changes produce plausible but wrong speech. | Compare phonemes and token IDs before comparing audio. |
| R8 | CLI quality scaffolding is lost during model work. | Keep template quality gate green and retain its tests/output/logging conventions. |
| R9 | Long text exceeds model or memory limits. | Define a supported length, reject or explicitly sentence-split, and defer batching. |
| R10 | GPU support becomes a second project. | Ship CPU parity first; make CUDA/WGPU a capability with evidence, not a promise. |
| R11 | Model preparation is not actually offline-friendly. | Store complete manifests and receipts; distinguish download-required from ready. |
| R12 | Terraform accidentally publishes or exposes more than the model archive. | Narrow bucket/object scope, reviewed plan, no credentials in code, and explicit public-read decision. |
| R13 | Multipart ETag is mistaken for the archive SHA-256. | Record the local SHA-256 in the source manifest and independently verify size/hash after download. |
| R14 | The 343 MB upload is too large for a simple single-request tool. | Use an S3-compatible multipart path; do not make Wrangler the required uploader. |
| R15 | Rehosting the weights or voice is not authorized. | Treat publication as blocked until provenance and redistribution permission are documented. |
| R16 | A common tensor abstraction forces unnecessary host copies or prevents backend-specific optimization. | Keep the public contract at prepared inputs/final audio and test backend-owned workspace lifecycles in W14. |
| R17 | LibTorch works locally only because a Python package supplies native DLLs and matching headers. | Make the runtime package/optional-accelerator boundary explicit and test a clean process in W15. |
| R18 | Vulkan compute dispatches successfully but misses Tensor Cores or loses to CUDA because of poor layouts, synchronization, or shader compilation overhead. | Require a 4090 microbenchmark and GPU timestamps before expanding the backend in W17-W18. |
| R19 | Fixed-shape 4090 tuning becomes an accidental cross-device support promise. | Query capabilities, mark unsupported devices, and keep the target matrix explicit in W17-W19. |
| R20 | Automatic backend selection chooses a fast but numerically wrong candidate or trusts stale evidence. | Gate receipts on parity, model/device/backend hashes, and selection-policy tests in W16 and W19. |
| R21 | Burn fusion/autotune changes duration rounding or output shape even when waveform values look plausible. | Keep fusion out of `auto` until sample-count and waveform parity pass; retain the failing receipt and investigate operation ordering/precision before relaxing the gate. |
| R22 | Burn tch's generated LibTorch bridge is tied to a newer LibTorch ABI than the upstream TorchScript environment. | Keep `burn-tch` distinct from direct LibTorch, record the isolated 2.9 toolchain in build evidence, and do not make the candidate part of automatic selection until its output-shape parity passes. |
| R23 | Burn's explicit Vulkan feature requests `cubecl-spirv` 0.8.1, which is not published to crates.io. | Pin the matching CubeCL v0.8.1 source family in Cargo, keep Burn WGPU and Ash Vulkan as separate paths, and retain the explicit RTX-4090 Vulkan support boundary. |
| R24 | Doctor output leaks a token, credential, or secret-bearing environment value through a renderer. | Use safe report projections, keep secrets out of report types, annotate internal values with `#[facet(sensitive)]`, and scan text/JSON/CSV fixtures. |
| R25 | Doctor checks drift from the behavior used by synthesis. | Reuse config/model-registry/runtime discovery functions and include a deep load/smoke path that exercises the real tch runtime. |
| R26 | Network or native checks hang, making the diagnostic command itself unreliable. | Bound every external probe, provide `--offline`, distinguish unavailable optional capabilities, and test failure timeouts. |
| R27 | A report schema change breaks LLM-assisted troubleshooting or scripts. | Include a schema version, keep stable check IDs/statuses, document format behavior, and add JSON contract tests. |

## Intent audit

### Extraction pass

Completed 2026-08-06. The new request was reduced to T1-T12. The audit
explicitly checked the repository rename, exact say syntax, model list,
model prepare, downloadable binary, local assets, Rust, Burn, GLaDOS
conversion, teamy-rust-cli grounding, and focused CLI scope.

### Traceability pass

Completed 2026-08-06. Each requirement maps to the product boundary, CLI
contract, design gates, work items, and acceptance matrix. The important
upstream-specific fact that the phonemizer is a model dependency is represented
in G2, W6, A5, and R2.

### Adversarial omission pass

Completed 2026-08-06. Checked that the plan does not:

- confuse TorchScript files with Burn records;
- omit the DeepPhonemizer checkpoint;
- assume the GPU vocoder is the only runtime;
- treat generated WAV output as proof of model parity;
- embed a 250 MB model package in the executable by default;
- confuse the MIT code license with model or voice redistribution rights;
- make the CLI depend on G:\ml\glados-tts-upstream;
- silently retain the upstream Flask/server architecture;
- promise real-time or accelerated performance without measurements.
- conflate Terraform bucket creation with successful object publication;
- treat an R2 object ETag as the archive's SHA-256;
- publish models.zip before the source/license decision is recorded.

#### W25 [x] Pin the final native Torch/CUDA toolchain

Work: Build a small compatibility probe for the selected latest published
`tch`/`torch-sys` 0.24.0 family and matching LibTorch 2.11.x runtime, then
record the exact compiler/CUDA/runtime packaging versions needed by the final
single-backend process. Verify that `tch::CModule` and
`tch::IValue::GenericDict` replace the handwritten C++ bridge. Keep the old
2.0.1 bridge available as a temporary oracle, but do not load it beside the
selected runtime. Burn/CubeCL upgrades are explicitly out of scope.

Validation: The selected stack must build from a clean environment, create and
operate CUDA tensors through `tch`, load the GLaDOS artifacts, pass a short
waveform correctness receipt, and produce a warm long-form RTX 4090 benchmark.
Do not advance to PyTorch 2.13 or an unreleased `tch-rs` revision as part of
this slice.

Evidence: [DEPENDENCIES.md](DEPENDENCIES.md). The final pair compiles and the
CUDA-link anchor makes `tch::Cuda::is_available()` report one RTX 4090 device.
The short correctness-gated receipt is 57 ms median / 62 ms p95 after two
warmups, with 2,590 ms model load and 26,880 generated samples.

#### W26 [x] Reshape `main` into the single tch runtime

Work completed: Removed Burn, CubeCL, Ash/Vulkan, WGPU, backend-comparison CLI,
handwritten TorchScript bridge, and obsolete Burn conversion/verification
examples from `main`. Replaced the runtime with direct Rust `tch::CModule` and
`tch::IValue` loading, kept the frontend model in `tch::nn`, added the
Windows-only CUDA import anchor, and changed the native bundle contract to
`glados-new.pt`, `vocoder-gpu.pt`, `glados-phonemizer.pt`, `frontend.tsv`, and
the two voice embeddings. The new `benchmark` command emits cold-load and
warm-latency JSON evidence.

Validation: `cargo check --all-targets`, `cargo test --all-targets`, model
preparation, correctness-gated RTX 4090 benchmarking, and an adjacent-DLL
package rehearsal pass with LibTorch 2.11.0+cu128 on the MSVC toolchain. The
generated tch-native archive is
`5fc80b76584ef7c078a417fb53e09fa8477b211e26458ad1ee8f4a25cf626e0f` and is
ready for the next Cloudflare publication step.

#### W27 [!] Publish and remotely rehearse the external model and native-runtime distribution

Work: Publish the tch-native prepared bundle through the Terraform-managed
R2 path, acquire it into an empty cache, and rehearse the installed executable
with only its documented adjacent LibTorch/CUDA runtime files. Keep model
artifacts independently updateable from the executable. This work is awaiting
explicit operator authorization and a recorded model/voice redistribution
rights decision; no Terraform, Cloudflare, DNS, credential, or remote
publication action is part of the current slice.

Validation:

```pwsh
terraform fmt -check -recursive infra\cloudflare
terraform validate infra\cloudflare
cargo test --all-targets
```

Completion criteria: a published-object verification, clean remote acquisition,
and installed-process receipt prove that the executable does not depend on the
upstream checkout, hidden Python, or per-invocation environment setup. The
local half of this criterion is recorded under W34; the remote half remains
open.

#### W28 [x] Define the typed doctor report and shared output integration

Work: Add a `doctor` command returning a typed `CliOutput` value through
`CliOutput::facet_with_csv(DoctorReport, ...)`.
Derive `Facet` report types with stable kebab-case names and explicit check
status/severity, summary, evidence, and remediation fields. Reuse the
teamy-rust-cli `GlobalArgs.output_format` and top-level single-emission path;
do not add a doctor-local formatter. Decide and document report schema
versioning and exit behavior while preserving the report when checks fail.

Keep raw credentials, access tokens, and secret environment values out of the
report model. Use `#[facet(sensitive)]` on internal/configuration values where
it provides defense in depth, but rely on safe projection rather than
serializer redaction.

Validation:

```pwsh
cargo test --all-targets
cargo run -- --output-format text doctor
cargo run -- --output-format json doctor
cargo run -- --output-format csv doctor
```

Completion notes: `src/cli/doctor/doctor_cli.rs` defines versioned
`DoctorReport`, `DoctorCheck`, and `DoctorStatus`; `src/cli/output.rs` keeps
the format decision at the top level and supports the flat CSV projection.
`cargo test --all-targets` includes text/JSON/CSV secret-safety coverage, and
manual text, JSON, and CSV invocations all rendered successfully.

#### W29 [x] Implement local configuration and model-cache checks

Work: Report the effective configuration sources and precedence without
revealing secret values; inspect model registry identity, prepared directory,
manifest, required artifact presence, byte sizes, hashes, formats, and
frontend/voice sidecars. Reuse existing config and artifact validation paths
so doctor and synthesis cannot disagree about whether a model is prepared.

Validation: Exercise empty-cache, valid-cache, stale-manifest, missing-file,
hash-mismatch, and environment-override fixtures through focused tests and a
real `cargo run -- --output-format json doctor` invocation. The command must
describe the next safe CLI action without performing it.

Completion notes: configuration values are reported with safe precedence
labels (`environment-override`, `remembered-config`, or default), while model
inspection reuses the catalog and prepared-manifest validator. Shallow mode
checks manifest/size state; deep mode verifies SHA-256 hashes. A current
installed run reported the prepared six-artifact bundle valid without creating
or changing files.

#### W30 [x] Implement native-runtime, CUDA, audio, and model-server checks

Work: Add bounded checks for LibTorch discovery/version/linkage, CUDA device
availability and identity, native DLL/runtime compatibility, output-directory
writability, audio playback capability, and configured public model-server
reachability. Separate cheap default checks from `doctor --deep`, which may
load the actual modules and perform a short synthesis smoke test. Add
`--offline` or equivalent skip semantics for network-dependent checks.

Checks must distinguish required failures from optional acceleration or
playback warnings, include actionable remediation commands, and never mutate
configuration, cache contents, or the network state.

Validation: Run shallow and deep doctor checks against the current RTX 4090
installation, an empty cache, a deliberately unavailable model URL, and a
process without the expected native runtime. Verify bounded failure behavior,
redacted output, and no created model/audio artifacts from diagnostics except
explicitly scoped temporary smoke-test files.

Completion notes: the command now reports tch/LibTorch linkage, CUDA/cuDNN
availability and runtime versions, Windows wave-output device availability,
output-path safety, and bounded HTTPS model-server probes. `--offline`
produces explicit skips; unconfigured R2D2FISH-OneDrive endpoints are explicit
skips; reachable/unreachable Teamy endpoints produce independent checks.
`doctor --deep --offline` passed the actual synthesis smoke test with 26,880
samples on the RTX 4090. An online run completed with bounded endpoint
failures rather than hanging in the current environment.

#### W31 [x] Document and acceptance-test the diagnostic surface

Work: Document the command examples, output-format behavior, report schema,
exit statuses, check depth, offline behavior, secret-redaction boundary, and
the fact that doctor diagnoses rather than repairs. Add CLI/integration tests
for the public contract and update the release/distribution rehearsal to use
doctor as its first troubleshooting step.

Validation:

```pwsh
cargo fmt --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo run -- --help
cargo run -- --output-format json doctor | ConvertFrom-Json
```

Completion notes: README documents shallow/deep/offline behavior, safe
versioned JSON, global output formats, and non-mutating diagnosis. `--help`
contains `doctor`; redirected JSON parsed with PowerShell; `cargo fmt`,
`cargo test --all-targets`, and `cargo clippy --all-targets --all-features --
-D warnings` pass with the pinned LibTorch build environment.

#### W32 [x] Make playback memory-backed and phoneme input explicit

Work: Keep generated audio in memory for `say` and `interactive` playback;
write WAV files only when an output destination is explicitly requested. Make
`write` require `--output` or `--output-dir`. Add a `--phonemes` mode that
validates GLaDOS's IPA-like symbol inventory and bypasses text normalization
and neural phonemization, while retaining ordinary English as the default.

Validation: `cargo fmt --check`, `cargo test --all-targets`, and
`cargo clippy --all-targets --all-features -- -D warnings` pass. The frontend
unit tests cover direct `eɪ` tokenization and unsupported-symbol rejection;
the output-path tests prove that no destination resolves to a persistent file.
Windows playback uses `PlaySoundW` with `SND_MEMORY | SND_SYNC`, so the WAV
buffer remains alive for the complete synchronous playback call.

#### W33 [x] Expose the text frontend through `phonemize`

Work: Add a `phonemize` command that uses the same prepared dictionary and
neural phonemizer as synthesis, reports the exact GLaDOS phoneme-symbol
sequence and integer token IDs, and loads only the text-side artifacts. Keep
the command on the shared typed text/JSON/CSV output path.

Validation: `cargo run -- phonemize "The letter A"` reports
`ðə lɛtɝ ə.` and the JSON form reports the same sequence with token IDs.
Frontend tests prove that the reported sequence tokenizes identically to
ordinary synthesis input.

#### W34 [x] Complete the local clean-machine distribution rehearsal

Work: Add `tools/rehearse-distribution.ps1` as a non-publishing rehearsal
harness. Stage the installed executable with its adjacent LibTorch/CUDA DLLs,
copy and hash the catalogued native archive, prepare it into an empty cache,
persist and override configuration, and run only the staged executable with a
cleared environment containing the package and Windows loader paths. Record a
versioned JSON receipt plus per-command stdout/stderr logs.

Validation: The 2026-08-23 receipt at
`target/distribution-rehearsal/20260823-203754580/receipt.json` contains 24
passing assertions and 13 command records. It verifies the native archive
(`217016604` bytes,
`5fc80b76584ef7c078a417fb53e09fa8477b211e26458ad1ee8f4a25cf626e0f`) and raw
archive (`343345374` bytes,
`afb60dd8944934ea5c67bd85de70f424c151b5f41b50dc039578716364fa68c4`), the
2.11.0+cu128 runtime manifest with 35 adjacent DLLs and no Python runtime,
typed deep doctor output, remembered and environment configuration provenance,
correctness-gated cold/warm benchmarking, in-memory `say`, explicit WAV
writing, two-line resident `interactive`, and expected missing-model,
corrupt-manifest, and missing-`torch_cuda.dll` failures. The child process
environment contained no development checkout, Python, Hugging Face
credentials, or remote model source.

#### W35 [x] Add bounded silent playback control

Work: Add `--volume <0.0..=1.0>` to `say` and `interactive`. Validate that the
value is finite and within the inclusive unit range, scale the retained PCM
samples before WAV encoding, and keep the normal synchronous in-memory
playback path intact. This permits end-to-end playback tests at volume zero
without skipping synthesis, WAV construction, or the operating-system audio
call.

Validation: Unit tests cover zero scaling and invalid values. The W34 receipt
executes both `say --volume 0` and `interactive --volume 0`; each completed
with no persistent output, logged the applied zero multiplier, and returned a
successful synchronous playback result.

## Next safe implementation slice

1. Resolve model/voice redistribution rights and obtain explicit authorization
   before attempting W27's remote publication or acquisition steps. Until then,
   rerun W34 locally when the package or model manifest changes.

Do not begin by manually rewriting every neural layer. First freeze the
reference tensors, model variants, frontend behavior, and artifact provenance.

## Completion rule

Planning is complete when G1-G19 have owners and decisions, W3 has produced
reference receipts, and the conversion boundary plus backend contract are
executable. The multi-backend and installed-configuration goal is complete
only when A1-A30 have evidence or explicit documented non-claims; the doctor
goal additionally requires A31-A35, which now have implementation and current
machine evidence. The first release may still exclude an
optional backend if its support boundary and evidence are honest, but it must
provide the diagnostic surface for the supported external-runtime path.
