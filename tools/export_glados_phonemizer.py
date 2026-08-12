"""Export DeepPhonemizer's forward Transformer for the native Burn runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def split_attention_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if tensor.shape[0] != 3 * 512:
        raise ValueError(f"expected a 3*512 attention tensor, got {tuple(tensor.shape)}")
    return tuple(tensor.split(512, dim=0))


def export(source: Path, output: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu")
    state = checkpoint["model"]
    exported: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.replace("encoder.layers.", "encoder.")
        key = key.replace("encoder.norm.", "norm.")
        if key.startswith("encoder.") and ".self_attn.in_proj_" in key:
            prefix, suffix = key.split(".self_attn.in_proj_", maxsplit=1)
            query, key_tensor, value_tensor = split_attention_tensor(value)
            exported[f"{prefix}.self_attn.query.{suffix}"] = query
            exported[f"{prefix}.self_attn.key.{suffix}"] = key_tensor
            exported[f"{prefix}.self_attn.value.{suffix}"] = value_tensor
        elif ".norm1." in key or ".norm2." in key or key.startswith("norm."):
            exported[key.replace(".weight", ".gamma").replace(".bias", ".beta")] = value
        else:
            exported[key] = value

    expected = 3 + 6 * 16 + 2 + 2
    if len(exported) != expected:
        raise ValueError(f"expected {expected} native tensors, got {len(exported)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    print(f"exported {len(exported)} tensors to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
