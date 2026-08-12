"""Regenerate or verify the Python frontend parity corpus.

This is a development-time oracle. It must be run with the upstream
repository and its virtual environment; the packaged Rust CLI does not import
this module or depend on Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def reference_cases(upstream_root: Path, inputs: list[str]) -> list[dict[str, Any]]:
    upstream_root = upstream_root.resolve()
    sys.path.insert(0, str(upstream_root))
    old_cwd = Path.cwd()
    os.chdir(upstream_root)
    try:
        from utils.text.cleaners import Cleaner
        from utils.text.tokenizer import Tokenizer

        cleaner = Cleaner("english_cleaners", True, "en-us")
        tokenizer = Tokenizer()
        cases = []
        for text in inputs:
            prepared = text if text[-1:] in ".?!" else f"{text}."
            cleaned = cleaner(prepared)
            cases.append(
                {
                    "input": text,
                    "cleaned": cleaned,
                    "token_ids": tokenizer(cleaned),
                }
            )
        return cases
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = json.loads(args.corpus.read_text(encoding="utf-8"))
    inputs = [case["input"] for case in expected["cases"]]
    actual = dict(expected)
    actual["cases"] = reference_cases(args.upstream_root, inputs)
    encoded = json.dumps(actual, ensure_ascii=False, indent=2) + "\n"

    if args.write:
        args.write.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")

    if args.check and actual["cases"] != expected["cases"]:
        raise SystemExit("frontend reference corpus does not match the upstream oracle")


if __name__ == "__main__":
    main()
