"""Inspect upstream GLaDOS model artifacts for the Rust/Burn converter.

This is a development-time tool. The packaged teamy-tts executable does not
import Python, PyTorch, or TorchScript; it consumes the Burn-native artifacts
produced by a later conversion step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": tensor.numel(),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def inspect_torchscript(path: Path) -> dict[str, Any]:
    module = torch.jit.load(str(path), map_location="cpu")
    state = module.state_dict()
    methods = sorted(module._c._method_names())
    operations: dict[str, int] = {}
    for method_name in methods:
        graph = module._c._get_method(method_name).graph
        for node in graph.nodes():
            operations[node.kind()] = operations.get(node.kind(), 0) + 1

    return {
        "kind": "torchscript",
        "methods": methods,
        "operations": dict(sorted(operations.items())),
        "state": {
            key: tensor_summary(value)
            for key, value in sorted(state.items())
            if isinstance(value, torch.Tensor)
        },
        "parameter_count": sum(
            value.numel() for value in state.values() if isinstance(value, torch.Tensor)
        ),
    }


def inspect_torch_file(path: Path) -> dict[str, Any]:
    try:
        module = torch.jit.load(str(path), map_location="cpu")
    except RuntimeError:
        value = torch.load(str(path), map_location="cpu")
        if isinstance(value, torch.Tensor):
            return {"kind": "tensor", "tensor": tensor_summary(value)}
        if isinstance(value, dict):
            return {
                "kind": "dictionary",
                "keys": sorted(str(key) for key in value),
                "tensor_values": {
                    str(key): tensor_summary(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                    if isinstance(item, torch.Tensor)
                },
            }
        raise TypeError(f"unsupported Torch value in {path}: {type(value)!r}")
    del module
    return inspect_torchscript(path)


def inspect_file(path: Path) -> dict[str, Any]:
    record = {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    record.update(inspect_torch_file(path))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    files = sorted(path for path in models_dir.rglob("*.pt") if path.is_file())
    inventory = {
        "format": "teamy-tts-upstream-model-inventory",
        "schema_version": 1,
        "models_dir": models_dir.as_posix(),
        "files": [inspect_file(path) for path in files],
    }
    encoded = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
