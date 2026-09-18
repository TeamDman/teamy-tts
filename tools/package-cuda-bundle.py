#!/usr/bin/env python3
"""Build the deterministic FP32 runtime bundle; no Torch/Python is needed to load it."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

EXPECTED = {
    "weights.safetensors": "f3ab504f0fb9f360c05d8ffd044cdea3f1ca00666fd51f8762031e3baf77b87b",
    "frontend.tsv": "2e229f9322839623f02a7f1bf0ce2b4f45ff56e67ebb32d27c3a069dcf4bcc69",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for name, expected in EXPECTED.items():
        path = args.model_dir / name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"{name}: expected validated FP32 artifact {expected}, got {actual}")
        files[name] = {"bytes": path.stat().st_size, "sha256": actual}
    manifest = {"schema": "teamy-glados-native-v1", "sample_rate": 22050,
                "precision": "float32", "voices": ["p1", "p2"], "files": files}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as archive:
        # An explicit allowlist excludes exporter fixtures and TorchScript sources.
        for name in sorted([*EXPECTED, "manifest.json"]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            data = manifest_bytes if name == "manifest.json" else (args.model_dir / name).read_bytes()
            archive.writestr(info, data, compresslevel=6)
    files["manifest.json"] = {"bytes": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    catalog = {"schema": "teamy-glados-native-v1", "sample_rate": 22050,
               "archive_bytes": args.output.stat().st_size, "archive_sha256": digest(args.output), "files": files}
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.catalog.with_name(args.catalog.stem + "-manifest.json").write_bytes(manifest_bytes)
    print(json.dumps(catalog, sort_keys=True))


if __name__ == "__main__":
    main()
