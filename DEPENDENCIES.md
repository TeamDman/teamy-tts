# Final single-backend dependency pinning

Status: tch-only migration in progress; final stable tch-rs family selected;
matching LibTorch package and GPU acceptance run pending.

This document is the dependency checkpoint for removing the backend comparison
implementations from `main`. `backend-comparison` preserves the historical
matrix; `main` is being reduced to one native Torch runtime.

## Current local inventory

| Layer | Current pin or installation | Role | Decision status |
|---|---|---|---|
| Rust | `rustc`/`cargo` 1.96.0, `x86_64-pc-windows-msvc` | Host toolchain | Keep for the first single-backend slice |
| GPU | GeForce RTX 4090, driver 610.88 | Target device | Keep; target-specific optimization remains intentional |
| CUDA toolkit | 13.3 (`nvcc` 13.3.73) | Optional native extension/kernel work | Keep installed; not a product runtime dependency |
| `tch` | 0.22.0 in the temporary local probe | Rust wrapper over LibTorch | Replace with the selected final line |
| `torch-sys` | 0.22.0 in the temporary local probe | Low-level FFI used by `tch` | Replace with the same exact `tch` family revision |
| LibTorch/PyTorch | 2.11.0+cu128 | Final Windows runtime package; CUDA link anchor required for MSVC | Accepted for the RTX 4090 probe |
| Direct TorchScript runtime | 2.0.1+cu117 | Historical handwritten C++ bridge | Removed from `main`; retained only in `backend-comparison` |

The two Torch installations are not interchangeable. The 2.0.1 environment
belongs to the upstream historical bridge; the 2.9 environment was useful for
the temporary tch migration probe. The final process must load one LibTorch
family, and the model files themselves remain independent of that runtime.

## Upstream versions relevant to the decision

- The latest published `tch` release is 0.24.0, with matching
  `torch-sys` 0.24.0 and the LibTorch 2.11 line.
- The `tch-rs` repository main branch may contain a newer, not-yet-published
  compatibility line. We will not use an unreleased git revision merely to
  chase the newest PyTorch release; a deliberate git pin can be considered
  later as a separate experiment.
We will not spend a dependency-upgrade cycle on Burn. The Burn/CubeCL stack is
comparison history and is no longer a dependency of `main`.

## Proposed final target

The selected target is:

```text
Rust 1.96 MSVC
    + tch 0.24.0
    + torch-sys 0.24.0
    + matching LibTorch 2.11.x CUDA build
    + local CUDA toolkit 13.3 for native extension/kernel compilation
    + RTX 4090-specific runtime tuning
```

This intentionally leaves PyTorch 2.13 out of the product pin. The release
rule is “newest stable published tch-rs family,” not “newest PyTorch.” That
keeps the Rust API, generated `torch-sys` bindings, LibTorch headers, and
runtime on one known release line while accepting that PyTorch itself may be
one release ahead.

We must first build a small probe that creates CUDA tensors, performs the
operations needed by GLaDOS, loads the required TorchScript/module format, and
runs a short reference sentence. The Rust `tch::CModule`/`IValue` API is
expected to replace our handwritten bridge: it supports named JIT methods and
`GenericDict` values, which covers the current `generate_jit` result
extraction.

We should not use PyTorch 2.14 nightly for the product pin. It can be a
separate performance experiment after the stable path is reproducible.

## CUDA decision

No CUDA SDK change is required merely because the product moves from the
current comparison stack to `tch`. LibTorch supplies its runtime libraries;
the local toolkit is needed for building native CUDA extensions and any
handwritten kernels. The installed 13.3 toolkit is already present and the
current Rust CUDA selector intentionally names 13.0 because `cudarc` 0.17.8
does not name 13.3.

The compatibility spike must answer two concrete questions before we change
the selector or install anything:

1. Does the selected LibTorch package load and execute correctly with the
   610.88 driver on the RTX 4090?
2. Do our custom C++/CUDA extension build flags and the selected `tch` native
   handles work with CUDA 13.3 without a copy or a second runtime?

The final product uses `tch` plus the native LibTorch runtime. If a genuinely
missing Torch operation requires a custom C++/CUDA operator, it will link
against the same selected LibTorch runtime and accept `tch` tensor handles;
that is an extension, not a second tensor backend.

## Pinning gates

Before stripping the comparison code, record all of the following in the
repository and in the release build:

1. Exact LibTorch archive/version and CUDA variant.
2. Exact `tch` and `torch-sys` source revision and their generated API line.
3. Exact CUDA toolkit/compiler version used for custom extensions.
4. MSVC/C++ standard and linker settings.
5. A clean-environment load test and short waveform correctness receipt.
6. A warm long-form benchmark receipt on the RTX 4090.

The direct 2.0.1 bridge is no longer compiled by `main` and must not be loaded
into the same process as the final LibTorch runtime.

On Windows MSVC, `main` carries a one-symbol CUDA import anchor because the
published `tch 0.24`/`torch-sys 0.24` build can otherwise have the linker drop
the unused `torch_cuda.dll` import. This is a loader/linker workaround only;
all tensors and operators still come from `tch`/LibTorch. The upstream fix is
being developed for a later tch-rs line.

## Temporary migration evidence

With `tch 0.24.0` and LibTorch 2.11.0+cu128, the direct Rust `tch::CModule`
path loaded the upstream graphs, detected one CUDA device, and passed the
finite/stable waveform gate plus canonical sample-count check on the RTX 4090:

```text
workload: Hello, friend
model_load_ms: 2590
warmups: 2
measurements: 5
warm measurement_ms: 56, 57, 57, 60, 62
median_ms: 57
p95_ms: 62
sample_count: 26880
audio_duration_ms: 1219
correctness_passed: true
correctness_gate: finite, stable waveform plus canonical Hello, friend sample-count check (26880)
```

The earlier `tch 0.22`/PyTorch 2.9 CPU result (232 ms median, 253 ms p95) is
retained only as migration history; it is no longer the product benchmark.
