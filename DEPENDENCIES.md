# Final single-backend dependency pinning

Status: inventory complete; stable tch-rs family selected; compatibility spike pending.

This document is the dependency checkpoint before removing the backend
comparison implementations from `main`. It intentionally separates the
current comparison matrix from the proposed final native Torch runtime.

## Current local inventory

| Layer | Current pin or installation | Role | Decision status |
|---|---|---|---|
| Rust | `rustc`/`cargo` 1.96.0, `x86_64-pc-windows-msvc` | Host toolchain | Keep for the first single-backend slice |
| GPU | GeForce RTX 4090, driver 610.88 | Target device | Keep; target-specific optimization remains intentional |
| CUDA toolkit | 13.3 (`nvcc` 13.3.73) | CubeCL and native CUDA compilation | Keep installed; do not change before the compatibility spike |
| CUDA Rust selector | `cudarc` 0.17.8, `CUDARC_CUDA_VERSION=13000` | Burn/CubeCL CUDA loading | Comparison-only until we decide whether Rust CUDA remains in the product |
| Burn | 0.19.1 | Current model structure and comparison backends | Remove from the final runtime unless the Torch migration needs it temporarily |
| CubeCL | 0.8.1, with matching shared crates patched from the v0.8.1 tag | Custom Burn CUDA kernel | Remove with the Burn comparison path |
| Burn tch | 0.19.1 | Burn's LibTorch backend | Remove as a runtime abstraction; retain its useful native-operation findings |
| `tch` | 0.22.0 | Rust wrapper over LibTorch | Replace with the selected final line |
| `torch-sys` | 0.22.0 | Low-level FFI used by `tch` | Replace with the same exact `tch` family revision |
| LibTorch/PyTorch | 2.9.0+cu128 | Burn-tch comparison runtime | Do not carry forward without a deliberate pin decision |
| Direct TorchScript runtime | 2.0.1+cu117 | Existing handwritten C++ bridge | Retire from the final product path |

The two Torch installations are not interchangeable. The 2.0.1 environment
belongs to the upstream TorchScript files; the 2.9 environment belongs to the
current `tch`/Burn-tch build. The final process must load one LibTorch family.

## Upstream versions relevant to the decision

- PyTorch 2.13.0 is newer than the product dependency we are selecting.
- PyTorch 2.14 is the current development/nightly line and is not a product
  target.
- The latest published `tch` release is 0.24.0, with matching
  `torch-sys` 0.24.0 and the LibTorch 2.11 line.
- The `tch-rs` repository main branch may contain a newer, not-yet-published
  compatibility line. We will not use an unreleased git revision merely to
  chase the newest PyTorch release; a deliberate git pin can be considered
  later as a separate experiment.
- Burn 0.21.0 is current upstream stable, but it is not a product target.
  Its workspace still declares `tch`/`torch-sys` 0.22.0, so upgrading Burn
  would not solve the Torch-version problem we are trying to solve.

We will not spend a dependency-upgrade cycle on Burn. The Burn/CubeCL stack
is comparison history and can be removed once the native `tch` runtime has
passed the migration gates.

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

The final product is intended to use `tch` plus the native LibTorch runtime.
The comparison-only `cudarc`, CubeCL, Burn, Ash, and handwritten TorchScript
bridge dependencies should be removed after the probe and migration tests
pass. If a genuinely missing Torch operation requires a custom C++/CUDA
operator, it will link against the same selected LibTorch runtime and accept
`tch` tensor handles; that is an extension, not a second tensor backend.

## Pinning gates

Before stripping the comparison code, record all of the following in the
repository and in the release build:

1. Exact LibTorch archive/version and CUDA variant.
2. Exact `tch` and `torch-sys` source revision and their generated API line.
3. Exact CUDA toolkit/compiler version used for custom extensions.
4. MSVC/C++ standard and linker settings.
5. A clean-environment load test and short waveform correctness receipt.
6. A warm long-form benchmark receipt on the RTX 4090.

The direct 2.0.1 bridge may remain temporarily as a correctness oracle during
the migration, but it must not be loaded into the same process as the final
LibTorch runtime.
