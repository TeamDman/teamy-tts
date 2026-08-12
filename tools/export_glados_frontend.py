"""Export the upstream DeepPhonemizer dictionary for the native runtime.

This is a development-time conversion tool.  The packaged executable reads
the resulting UTF-8 TSV and never imports Python, PyTorch, or DeepPhonemizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    dictionary = checkpoint["phoneme_dict"]["en_us"]
    lines = [
        f"{word}\t{phonemes}"
        for word, phonemes in sorted(dictionary.items(), key=lambda pair: pair[0])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} dictionary entries to {args.output}")


if __name__ == "__main__":
    main()
