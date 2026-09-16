"""Development-only conversion of the deployed GLaDOS program to numerical state.

The output weights contain no executable graph. TorchScript source is recorded
separately as a local audit reference, never consumed by the native runtime.
Run with the upstream Python environment, using explicitly supplied paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def write_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write the documented safetensors format without a runtime Python package."""
    header = {"__metadata__": {"format": "teamy-glados-native-v1"}}
    chunks = []
    offset = 0
    types = {torch.float32: "F32", torch.int64: "I64", torch.float16: "F16"}
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        dtype = types[value.dtype]
        data = value.numpy().tobytes(order="C")
        header[name] = {"dtype": dtype, "shape": list(value.shape),
                        "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for data in chunks:
            stream.write(data)


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def frozen_vocoder_weights(module) -> dict[str, torch.Tensor]:
    # The deployed frozen program has an EMPTY state_dict: all 156 learned
    # tensors and its averaging scalar are in the constant table instead.
    _, constant_table = module.code_with_constants
    constants = constant_table.const_mapping
    if len(constants) != 157 or module.state_dict():
        raise ValueError("unsupported vocoder revision: expected the deployed frozen HiFiGAN")
    output = {}
    index = 0

    def conv(name, shape):
        nonlocal index
        w, b = constants[f"c{index}"], constants[f"c{index + 1}"]
        if list(w.shape) != shape:
            raise ValueError(f"{name}: {tuple(w.shape)} != {shape}")
        output[f"vocoder.{name}.weight"] = w
        output[f"vocoder.{name}.bias"] = b
        index += 2

    conv("conv_pre", [512, 80, 7])
    for stage, (input_ch, output_ch, kernel) in enumerate(
            [(512, 256, 16), (256, 128, 16), (128, 64, 4), (64, 32, 4)]):
        conv(f"ups.{stage}", [input_ch, output_ch, kernel])
        for block, size in enumerate([3, 7, 11]):
            for layer in range(3):
                for group in ("convs1", "convs2"):
                    conv(f"resblocks.{stage * 3 + block}.{group}.{layer}",
                         [output_ch, output_ch, size])
        if stage == 0:
            if constants[f"c{index}"].item() != 3.0:
                raise ValueError("unexpected residual branch averaging constant")
            index += 1
    conv("conv_post", [1, 32, 7])
    assert index == 157
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    acoustic = torch.jit.load(str(args.models_dir / "glados-new.pt"), map_location="cpu").eval()
    vocoder = torch.jit.load(str(args.models_dir / "vocoder-gpu.pt"), map_location="cpu").eval()
    weights = {}
    for prefix, module in [("acoustic", acoustic), ("vocoder", vocoder)]:
        weights.update({f"{prefix}.{k}": v for k, v in module.state_dict().items()})
    weights.update(frozen_vocoder_weights(vocoder))
    phonemizer = torch.load(args.models_dir / "glados-phonemizer.pt", map_location="cpu")
    weights.update({f"phonemizer.{k}": v for k, v in phonemizer.items()})
    for voice in ("p1", "p2"):
        weights[f"voice.{voice}"] = torch.from_numpy(
            np.fromfile(args.models_dir / f"voice-{voice}.f32le", dtype="<f4").copy()
        ).reshape(1, 256)
    write_tensors(args.output / "weights.safetensors", weights)
    shutil.copyfile(args.models_dir / "frontend.tsv", args.output / "frontend.tsv")
    inventory = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                 for name, value in sorted(weights.items())}
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    with (args.output / "torchscript-source.txt").open("w", encoding="utf-8") as stream:
        for prefix, model in [("acoustic", acoustic), ("vocoder", vocoder)]:
            for name, module in model.named_modules():
                for method in module._c._method_names():
                    stream.write(f"\n# {prefix}.{name}::{method}\n")
                    stream.write(str(module._c._get_method(method).code))
    # Known exact token fixture from the existing product's internal warmup.
    fixtures = [
        ("short-p2", [97, 24, 106, 27, 20, 5, 10, 79, 72, 28, 68, 54, 6], "p2", 1.0),
        ("short-p1", [97, 24, 106, 27, 20, 5, 10, 79, 72, 28, 68, 54, 6], "p1", 1.0),
        ("hello-fast", [97, 24, 106, 27, 20, 1], "p2", 1.25),
        ("long-p2", [97, 24, 106, 27, 20, 5, 10, 79, 72, 28, 68, 54, 6] * 8, "p2", 1.0),
    ]
    # Frozen tensor constants must be placed by the loader; Module.to() cannot
    # move them because they are not registered parameters/buffers.
    vocoder = torch.jit.load(str(args.models_dir / "vocoder-gpu.pt"), map_location=args.device).eval()
    records = []
    with torch.inference_mode():
        for name, token_values, voice, alpha in fixtures:
            tokens = torch.tensor([token_values], dtype=torch.long)
            speaker = weights[f"voice.{voice}"]
            output = acoustic.generate_jit(tokens, speaker, alpha)
            mel = output["mel_post"].contiguous()
            audio = vocoder(mel.to(args.device)).cpu().contiguous()
            tensors = {"tokens": tokens, "speaker": speaker, "mel": mel, "audio": audio}
            tensors.update({f"acoustic.{k}": v for k, v in output.items()
                            if isinstance(v, torch.Tensor)})
            write_tensors(args.output / f"{name}.safetensors", tensors)
            records.append({"name": name, "voice": voice, "alpha": alpha,
                            "tokens": token_values, "frames": mel.shape[-1],
                            "samples": audio.numel()})
        torch.manual_seed(1729)
        for prefix in ("dur_pred.rnn", "pitch_cond_pred.rnn", "pitch_pred.rnn",
                       "energy_pred.rnn", "prenet.rnn", "postnet.rnn", "lstm"):
            module = acoustic
            for part in prefix.split("."):
                module = getattr(module, part)
            inputs = weights[f"acoustic.{prefix}.weight_ih_l0"].shape[1]
            time = 105 if prefix in ("postnet.rnn", "lstm") else 13
            x = torch.randn(1, time, inputs) * 0.1
            y, _ = module._c._get_method("forward__0")(x, None)
            write_tensors(args.output / f"rnn-{prefix}.safetensors", {"input": x, "output": y})
    manifest = {"schema": "teamy-glados-native-v1", "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": args.device,
                "sample_rate": 22050, "fixtures": records,
                "weights": fingerprint(args.output / "weights.safetensors"),
                "sources": [fingerprint(args.models_dir / filename) for filename in
                            ["glados-new.pt", "vocoder-gpu.pt", "glados-phonemizer.pt",
                             "voice-p1.f32le", "voice-p2.f32le", "frontend.tsv"]],
                "export_seconds": perf_counter() - started}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tensors": len(weights), "fixtures": records,
                      "weight_bytes": manifest["weights"]["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
