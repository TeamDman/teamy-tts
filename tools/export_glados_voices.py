"""Export upstream speaker tensors as little-endian float32 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for voice in ("p1", "p2"):
        source = args.models_dir / "emb" / f"glados_{voice}.pt"
        tensor = torch.load(source, map_location="cpu").detach().cpu().contiguous()
        if tuple(tensor.shape) != (1, 256):
            raise ValueError(f"unexpected {source} shape: {tuple(tensor.shape)}")
        output = args.output_dir / f"voice-{voice}.f32le"
        output.write_bytes(tensor.numpy().astype("<f4").tobytes())
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
