"""Compare real release CLIs and upstream Python on a fixed, varied corpus.

All child processes run serially. Receipts include binary hashes, raw reports,
PCM parity, and external process elapsed time. No audio is played.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import time
import wave
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--libtorch", type=Path, required=True)
parser.add_argument("--cuda", type=Path, required=True)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--python", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--skip-python", action="store_true")
parser.add_argument("--case", action="append", help="Run selected case IDs")
args = parser.parse_args()
repo = args.repo.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
bins = {"original": repo / "target/baseline/release/teamy-tts.exe",
        "original-f32": repo / "target/baseline/release/teamy-tts.exe",
        "native": repo / "target/native-cli/release/teamy-tts.exe"}
environments = {}
for name in bins:
    env = os.environ.copy()
    for key in ("LIBTORCH", "TEAMY_TTS_BACKEND", "TEAMY_TTS_TORCH_DEVICE", "TEAMY_TTS_TORCH_MODEL_DIR",
                "NVIDIA_TF32_OVERRIDE", "GLADOS_RNN_ALGORITHM", "GLADOS_CUDA_POOL_LIMIT_MB"):
        env.pop(key, None)
    env["TEAMY_TTS_HOME_DIR"] = str(output / f"{name}-home")
    env["NO_COLOR"] = "1"
    if name.startswith("original"):
        env["PATH"] = str(args.libtorch / "lib") + os.pathsep + env["PATH"]
        env["TEAMY_TTS_TORCH_MODEL_DIR"] = str(args.models)
        env["TEAMY_TTS_MODEL_DIR"] = str(args.models.parent.parent)
        env["TEAMY_TTS_TORCH_DEVICE"] = "0"
        if name == "original-f32":
            env["NVIDIA_TF32_OVERRIDE"] = "0"
    else:
        env["PATH"] = os.pathsep.join([str(args.cuda / "bin"), str(args.cuda / "bin/x64"), env["PATH"]])
        env["GLADOS_CUDNN_LIBRARY"] = str(args.libtorch / "lib/cudnn64_9.dll")
        env["TEAMY_TTS_NATIVE_MODEL_DIR"] = str(repo / "artifacts/native-glados")
    environments[name] = env

def run(name, arguments, receipt):
    started = time.perf_counter()
    result = subprocess.run([str(bins[name]), "--output-format", "json", *arguments],
                            cwd=repo, env=environments[name], capture_output=True,
                            encoding="utf-8", timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    elapsed = (time.perf_counter() - started) * 1000
    (output / f"{receipt}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    (output / f"{receipt}.stdout.json").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{receipt}: exit {result.returncode}: {result.stderr[-2000:]}")
    return (json.loads(result.stdout) if arguments[0] != "write" else None), elapsed

def pcm(path):
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getframerate() == 22050
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64) / 32767.0

result = {"binaries": {k: {"sha256": hashlib.sha256(v.read_bytes()).hexdigest(), "bytes": v.stat().st_size}
                       for k, v in bins.items()}, "frontend": [], "cases": [],
          "boundary": "CLI benchmark text to host F32 PCM; model_load includes constructor warmup. External process total also includes all measurements and exit.",
          "filesystem_cache": "uncontrolled warm OS cache; not storage-cold"}

def save():
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

corpus = json.loads((repo / "reference/frontend-corpus.json").read_text(encoding="utf-8"))
for i, case in enumerate(corpus["cases"]):
    actual, _ = run("native", ["phonemize", case["input"]], f"frontend-{i}")
    assert actual["token-ids"] == case["token_ids"] and actual["phonemes"] == case["cleaned"]
    result["frontend"].append({"text": case["input"], "passed": True})
save()
print("Native frontend corpus: all four cases passed", flush=True)

cases = [
    ("short-p2", "Hello, friend", "p2", 1.0),
    ("short-p1", "Hello, friend", "p1", 1.0),
    ("hello-fast", "hello!", "p2", 1.25),
    ("neural", "Quux electroencephalographically.", "p2", 1.0),
    ("numbers", "Mrs. 42 costs $3.50.", "p2", 0.8),
    ("long", " ".join(["Hello, friend."] * 8), "p2", 1.0),
]
for case_id, text, voice, alpha in cases:
    if args.case and case_id not in args.case:
        continue
    row = {"id": case_id, "text": text, "voice": voice, "alpha": alpha, "benchmarks": {}}
    options = ["--voice", voice, "--alpha", str(alpha)]
    oracle = output / f"{case_id}-oracle.safetensors"
    exported = subprocess.run([str(args.python), str(repo / "native/tools/export_text_fixture.py"),
        "--upstream", str(args.upstream), "--models", str(args.models), "--text", text,
        *options, "--output", str(oracle)], cwd=repo, capture_output=True, encoding="utf-8",
        timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    if exported.returncode:
        raise RuntimeError(exported.stderr[-2000:])
    (output / f"{case_id}-oracle.json").write_text(exported.stdout, encoding="utf-8")
    data = oracle.read_bytes()
    header_size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8+header_size])
    lo, hi = header["audio"]["data_offsets"]
    expected = np.frombuffer(data[8+header_size+lo:8+header_size+hi], dtype="<f4").astype(np.float64)
    for name in bins:
        run(name, ["write", text, *options, "--output", str(output / f"{case_id}-{name}.wav")], f"{case_id}-{name}-write")
    actual = pcm(output / f"{case_id}-native.wav")
    same_shape = expected.shape == actual.shape
    parity = {"reference": "upstream torch 2.0.1 deployed model FP32 tensors", "same_shape": same_shape, "reference_samples": expected.size, "native_samples": actual.size}
    if same_shape:
        parity.update(relative_rms=float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)),
                      max_abs=float(np.max(np.abs(actual - expected))))
    parity["passed"] = same_shape and parity["relative_rms"] <= 0.001 and parity["max_abs"] <= 0.001
    row["pcm_parity"] = parity
    row["baseline_vs_oracle"] = {}
    for name in ["original", "original-f32"]:
        baseline_pcm = pcm(output / f"{case_id}-{name}.wav")
        metrics = {"same_shape": baseline_pcm.shape == expected.shape}
        if metrics["same_shape"]:
            metrics.update(relative_rms=float(np.linalg.norm(baseline_pcm - expected) / max(np.linalg.norm(expected), 1e-30)), max_abs=float(np.max(np.abs(baseline_pcm - expected))))
        metrics["passes_same_gate"] = metrics["same_shape"] and metrics["relative_rms"] <= 0.001 and metrics["max_abs"] <= 0.001
        row["baseline_vs_oracle"][name] = metrics
    result["cases"].append(row)
    save()
    if not parity["passed"]:
        raise RuntimeError(f"{case_id}: PCM parity failed: {parity}")
    print(f"{case_id}: WAV parity passed ({actual.size} samples)", flush=True)
    for repeat in range(args.repeats):
        # Alternate order to reduce a systematic thermal/cache order bias.
        for name in (list(bins) if repeat % 2 == 0 else list(reversed(bins))):
            report, elapsed = run(name, ["benchmark", text, *options, "--warmups", "3", "--measurements", str(args.iterations)],
                                  f"{case_id}-{name}-{repeat}")
            assert report["sample_count"] == actual.size and report["correctness_passed"]
            row["benchmarks"].setdefault(name, []).append({"report": report, "external_process_ms": elapsed})
            save()
            print(f"{case_id} {name} #{repeat}: load {report['model_load_ms']} ms, median {report['median_ms']} ms, p95 {report['p95_ms']} ms", flush=True)
    if not args.skip_python:
        destination = output / f"{case_id}-python.json"
        completed = subprocess.run([str(args.python), str(repo / "native/tools/benchmark_python.py"),
            "--mode", "upstream", "--upstream-root", str(args.upstream), "--text", text,
            *options, "--iterations", str(args.iterations), "--output", str(destination)],
            cwd=repo, capture_output=True, encoding="utf-8", timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW)
        (output / f"{case_id}-python.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(completed.stderr[-2000:])
        report = json.loads(destination.read_text(encoding="utf-8"))
        row["benchmarks"]["python"] = report
        row["python_same_sample_count"] = report["sample_count"] == actual.size
        save()
        print(f"{case_id} python: load {report['ready_from_python_entry_ms']:.1f} ms, median {report['median_ms']:.1f} ms", flush=True)
print("Comparison complete", flush=True)
