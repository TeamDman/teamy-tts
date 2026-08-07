# teamy-tts implementation plan

Status: active planning baseline; implementation has not started.
Plan owner: Teamy
Plan path: G:\Programming\Repos\teamy-tts\PLAN.md
Last updated: 2026-08-06
Current focus: [~] freeze the model contract and conversion strategy

This is the living work contract for turning the local GLaDOS TTS upstream
project into a downloadable Rust/Burn command-line application.

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

## User guidance ledger

| ID | Requirement or intent | Status | Traceability |
|---|---|---|---|
| T1 | Create a new teamy-tts repository. | Confirmed | Scope, W1 |
| T2 | Provide teamy-tts say --model glados "hello!" --output output.wav. | Confirmed | CLI contract, W10 |
| T3 | Provide teamy-tts model list. | Confirmed | CLI contract, W9-W10 |
| T4 | Provide teamy-tts model prepare glados; the braces in the request are treated as a placeholder for a model identifier. | Confirmed | CLI contract, W9-W10 |
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
published through a stable HTTPS download endpoint. The CLI must not contain
R2 access keys or depend on the S3 API for normal downloads.

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
6. Local Burn inference for the GLaDOS text frontend, ForwardTacotron, and
   HiFiGAN path selected for the release.
7. Mono 22050 Hz WAV output.
8. CPU reference backend and one accelerated backend only when parity and
   packaging evidence support it.
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
teamy-tts model prepare glados
teamy-tts say --model glados "hello!" --output output.wav
~~~

### Initial say options

- --model <id>, required;
- one positional text value, with a future --stdin path;
- --output <path>, required for the first release;
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

model prepare consumes an acquired raw archive, verifies its individual model
files, performs any required conversion or extraction into the Burn-native
runtime package, writes the prepared manifest atomically, and reports whether
the result is ready for say.

The command should be safe to rerun and should never treat a partially
downloaded directory as a prepared model.

## Architecture

~~~mermaid
flowchart LR
    C[figue/facet CLI] --> A[typed application actions]
    A --> R[model registry sources and doctor]
    A --> F[text normalization and phonemization]
    A --> B[Burn inference pipeline]
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

Model preparation must use temporary files and atomic rename, and must not
overwrite a verified revision in place.

## Design gates

| ID | Gate | Current position | Exit evidence |
|---|---|---|---|
| G1 | Model distribution source | Upstream README points to Google Drive; a release-hosting source is preferred. | Stable source, checksums, license record, and clean-device download rehearsal. |
| G2 | Text frontend runtime | Pure Rust/Burn is the intended target; feasibility is unproven. | Ordinary English text produces the same phoneme IDs as the Python oracle, or an explicit sidecar decision is documented. |
| G3 | TorchScript-to-Burn conversion format | Use deterministic extracted tensors/config, then Burn-native records. | Converter output and schema are versioned and reloadable without Python. |
| G4 | Model variant | Start with glados-new + p2 + one vocoder variant; p1 and alternatives are capabilities. | Variant manifest, parity corpus, and default selection decision. |
| G5 | Burn backend | CPU reference first; CUDA or WGPU only after parity and packaging checks. | Device matrix, output parity, and timing evidence. |
| G6 | Audio contract | Mono 22050 Hz WAV int16 for the first release. | WAV header and waveform tests plus a reference output. |
| G7 | Text/length contract | Short and medium English text first; sentence splitting and long text are explicit follow-up. | Length limits, failure behavior, and corpus coverage. |
| G8 | CLI foundation | Initialize from teamy-rust-cli and preserve its quality/logging/output conventions. | Cargo build, --help/--version, output formats, clippy, tests, and check-all.ps1. |
| G9 | Weight and voice redistribution | Upstream code is MIT; model/voice redistribution is unresolved. | Written release decision and included notices. |
| G10 | Parity threshold | Numerical thresholds must be established from reference runs, not guessed. | Per-stage tensor tolerances and final waveform criteria. |
| G11 | Performance target | Do not promise realtime until measured end to end. | Cold-start, warm-start, CPU/GPU latency, memory, and output-duration report. |
| G12 | R2 infrastructure and credentials | Terraform creates the bucket; credentials remain in environment/CI secrets; Terraform apply is deployment work, not an implicit local action. | terraform fmt/validate/plan, reviewed diff, apply receipt, and secret scan. |
| G13 | Archive publication | The Teamy object has an immutable versioned key, correct content type/cache policy, expected byte count, and verified SHA-256. | Multipart upload receipt plus independent HEAD/download verification. |
| G14 | Upstream source adapter | The exact R2D2FISH-OneDrive URL and its redirect/download behavior must be recorded. | Source manifest and one successful acquisition test. |

## Work breakdown

### Phase 1: CLI foundation and contract

#### W1 [~] Initialize from teamy-rust-cli

Work: Use the template initializer to create the Rust package in
G:\Programming\Repos\teamy-tts. Rename package metadata, app/cache variables,
repository URLs, Windows resources, and README. Retain figue/facet parsing,
shared output, logging, cancellation, version metadata, tests, and
check-all.ps1.

Validation: cargo run -- --help, cargo run -- --version, text/JSON/CSV output,
and the template quality gate run in the new repository.

Completion: A clean Rust CLI shell exists with no GLaDOS inference yet and no
template placeholders left in user-facing metadata.

#### W2 [ ] Freeze command and manifest schemas

Work: Define typed command arguments, model IDs, voice IDs, device IDs,
preparation states, output formats, error categories, manifest schema, and
structured receipt schema.

Validation: Argument fuzz/round-trip tests and JSON fixture round-trips cover
invalid combinations and unknown model revisions.

Completion: CLI parsing and model metadata are stable enough for the converter
and runtime to target.

### Phase 2: reference oracle and conversion tools

#### W3 [ ] Make the Python reference deterministic

Work: Extract a small reference runner from glados.py that accepts one text,
voice, alpha, and output path; emits intermediate token IDs, phonemes, mel
metadata, waveform statistics, timings, and model fingerprints.

Validation: Reference outputs are repeatable on the inspected environment;
empty input, punctuation, numbers, abbreviations, and unsupported characters
have explicit behavior.

Completion: The Python oracle can generate a machine-readable receipt and WAV
fixture for every parity test case.

#### W4 [ ] Inventory TorchScript model contracts

Work: Record module methods, graph operations, tensor names/shapes/dtypes,
constants, layer configuration, variant differences, and operator gaps for
glados-new, glados, and each vocoder.

Validation: The inventory is generated from the files and checked into an
artifact format; every required operation has a planned Burn equivalent or a
named blocker.

Completion: No conversion work depends on reverse-engineering an opaque .pt
file during Rust implementation.

#### W5 [ ] Build deterministic model extraction

Work: Extract state and configuration from TorchScript or its originating
PyTorch structures into a versioned interchange directory. Include embeddings,
phonemizer data, tensor layouts, and checksums.

Validation: Extract twice and compare manifests and tensor hashes. Reload all
extracted tensors in a small Python verifier.

Completion: Conversion inputs are reproducible and independent of the
developer's current working directory.

#### W5A [ ] Define raw model acquisition sources

Work: Define source descriptors for Teamy and R2D2FISH-OneDrive, including
display name, URL, archive key, expected byte count, SHA-256, content type,
source provenance, redirect policy, and whether resumable range downloads are
supported.

Validation: The supplied archive is checked against the Teamy source manifest;
the upstream-maintainer URL is tested without storing credentials; an invalid
source cannot be selected silently.

Completion: The CLI can distinguish raw archive acquired, raw archive
verified, and prepared model ready.

#### W5B [~] Add Terraform-managed Cloudflare R2 infrastructure

Work: Add an infra/cloudflare directory with pinned Cloudflare and AWS
provider versions, explicit account/bucket/location literals, an R2 bucket,
appropriate lifecycle behavior for incomplete multipart uploads, and only
outputs with a documented consumer. Keep all credentials out of the repository
and state inputs.

Validation: terraform fmt, terraform init, terraform validate, and
terraform plan run with placeholder-safe or test credentials. A review checks
that no public read policy or custom-domain change is applied accidentally.

Completion: The reviewed Terraform plan creates only the intended R2
infrastructure and can be applied by an authorized operator.

Current state (2026-08-06): `infra/cloudflare` now contains pinned Cloudflare
and AWS providers, explicit deployment literals, an Azure remote state
backend, the R2 bucket, native incomplete-multipart lifecycle configuration,
content-addressed archive publication, and credential instructions.
`terraform init`, `terraform fmt-check`, and `terraform validate` pass. The
existing local state was migrated to the private Azure statefiles container;
the remote state currently contains the bucket and archive object. The
lifecycle apply still needs to be completed after the 1Password CLI session
is authenticated.

#### W5C [~] Publish and verify models.zip

Work: Add a repeatable multipart publication script or CI task using an
S3-compatible R2 client, keyed by archive SHA-256. Upload the supplied
models.zip, attach content type/cache metadata, write a small public source
manifest, and verify the object with HEAD plus an independent checksum/size
check.

Validation: Re-run publication with the same archive, change a test archive,
interrupt a multipart upload, and verify that incomplete uploads are cleaned
up. Confirm the object can be downloaded through the exact Teamy HTTPS URL.

Completion: The Teamy source manifest points to an immutable verified object;
models.zip exists in the R2 bucket, and no R2 secret is present in the CLI or
repository.

Current state (2026-08-06): Terraform has uploaded the supplied archive to the
content-addressed R2 object key and recorded it in remote state. Independent
HEAD/download verification and the stable HTTPS source manifest remain
outstanding.

### Phase 3: Rust/Burn model implementation

#### W6 [ ] Implement the text frontend

Work: Port or convert the upstream cleaning, number expansion, abbreviation
handling, phonemization, IPA filtering, and symbol-to-ID mapping according to
G2. Keep a debug mode that prints the phoneme sequence and token IDs.

Validation: Compare every frontend stage with W3 fixtures. Include punctuation,
numbers, abbreviations, unicode/unidecode cases, empty input, and long input.

Completion: say can produce the exact token IDs expected by the oracle for the
supported English corpus without hidden upstream Python files.

#### W7 [ ] Implement ForwardTacotron in Burn

Work: Reconstruct the inference-only generate_jit path: pitch-condition
prediction, duration prediction with alpha, pitch and energy prediction,
length regulation, LSTM, mel projection, postnet, and mel_post padding.

Validation: Compare token-derived intermediate tensors and mel_post against
the Python oracle. Test p1/p2 embeddings and glados-new before older glados.

Completion: Burn produces mel_post within the documented intermediate
tolerances for the parity corpus on the reference backend.

#### W8 [ ] Implement HiFiGAN in Burn

Work: Reconstruct the selected vocoder variant, including transposed
convolutions, residual branches, leaky-ReLU, normalization/division, and final
tanh. Add a clear variant interface for CPU-LQ, CPU-HQ, and GPU-compatible
weights where supported.

Validation: Compare mel-to-waveform tensors with the Python oracle and check
waveform bounds, sample count, silence behavior, and output duration.

Completion: One documented vocoder variant generates parity-tested waveforms
from Burn mel_post without Python.

#### W9 [ ] Integrate model registry and preparation

Work: Define the known catalog, source manifests, raw archive acquisition,
checksum verification, staging, atomic install, cache resolution, and device
compatibility checks. Keep raw acquisition separate from Burn preparation.

Validation: Test fresh acquisition, repeated acquisition, interrupted download,
bad archive checksum, missing file, offline cache, unknown source, fresh
prepare, and incompatible variant.

Completion: model list, model show, model acquire-unprepared Teamy, and model
prepare glados work without loading inference and produce actionable
diagnostics.

### Phase 4: user-facing inference

#### W10 [ ] Connect say to local Burn inference

Work: Connect CLI parsing, model preparation checks, frontend, Burn model
loading, device selection, alpha/voice options, WAV writing, output
validation, structured logs, and receipt generation.

Validation: Run the exact target command on a clean prepared cache and compare
the output against the Python reference. Test output paths, errors,
cancellation, and repeated invocations.

Completion: teamy-tts say --model glados "hello!" --output output.wav works
without the upstream checkout or Python runtime.

#### W11 [ ] Package and rehearse downloadability

Work: Build release artifacts, include notices and model preparation docs,
record version/Git metadata, and document app/cache overrides, device
selection, model sources, and troubleshooting.

Validation: Rehearse on a clean Windows environment with no upstream checkout,
no pre-existing cache, and no hidden PATH dependency. Run check-all.ps1 and
the acceptance matrix.

Completion: A new user can download the executable, run model list, prepare
glados, run say, and locate a valid WAV.

### Phase 5: evidence and follow-up

#### W12 [ ] Establish parity, performance, and release evidence

Work: Maintain a corpus and acceptance matrix covering frontend token IDs,
mel_post, waveform statistics, perceptual/audio comparisons, cold/warm
latency, memory, backend, model revision, and device.

Validation: Reports identify evidence type, tool versions, fixture scope,
thresholds, failures, and non-claims. Do not label sampled parity exhaustive.

Completion: Release claims are traceable to receipts and measurements.

#### W13 [ ] Add deferred capabilities only after the first release

Work: Consider stdin, multiple output formats, streaming/playback, p1 as a
first-class voice, CPU-HQ/GPU variants, sentence batching, and a local service.

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
| A4 | say accepts the requested command shape. | CLI round-trip test and end-to-end invocation. |
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

## Next safe implementation slice

1. Run the teamy-rust-cli initializer into G:\Programming\Repos\teamy-tts.
2. Replace template metadata and retain its quality/testing foundation.
3. Implement W2 schemas and a model-catalog-only model list.
4. Add W5A's source manifest for the supplied archive without uploading it.
5. Complete the W5B lifecycle apply after authenticating the 1Password CLI,
   then perform W5C's independent object verification before recording a
   public-download decision.
6. Build W3's deterministic Python reference receipt before writing Burn
   layers.

Do not begin by manually rewriting every neural layer. First freeze the
reference tensors, model variants, frontend behavior, and artifact provenance.

## Completion rule

Planning is complete when G1-G14 have owners and decisions, W3 has produced
reference receipts, and the conversion boundary is executable. The first
release is complete only when A1-A16 have evidence or explicit documented
non-claims.
