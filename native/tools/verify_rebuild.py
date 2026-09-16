"""Tie a rebuilt executable to an existing paired matrix after source cleanup.

Require byte-identical WAV output and rerun every native workload. Preserve the
original matrix, and report this build's hashes and measurements separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--matrix", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
binary = args.binary.resolve()
matrix = json.loads((args.matrix / "comparison.json").read_text(encoding="utf-8"))
args.output.mkdir(parents=True, exist_ok=True)
def run(arguments, label):
    completed = subprocess.run([str(binary), "--output-format", "json", *arguments],
        capture_output=True, encoding="utf-8", timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    (args.output / f"{label}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.output / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    return completed.stdout
result = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
          "paired_matrix_binary_sha256": matrix["binaries"]["native"]["sha256"], "cases": [],
          "gate": "byte-identical WAV to previously validated native output; new median, p95 and model load below each baseline's median across repeats"}
for case in matrix["cases"]:
    options = [case["text"], "--voice", case["voice"], "--alpha", str(case["alpha"])]
    wav = args.output / (case["id"] + ".wav")
    run(["write", *options, "--output", str(wav.resolve())], case["id"] + "-write")
    assert wav.read_bytes() == (args.matrix / f"{case['id']}-native.wav").read_bytes(), case["id"]
    report = json.loads(run(["benchmark", *options, "--warmups", "3", "--measurements", "20"], case["id"]))
    gates = {}
    for baseline in ("original", "original-f32"):
        for metric in ("model_load_ms", "median_ms", "p95_ms"):
            threshold = statistics.median(float(r["report"][metric]) for r in case["benchmarks"][baseline])
            gates[f"{baseline}-{metric}"] = float(report[metric]) < threshold
    python = case["benchmarks"].get("python")
    if python:
        for metric in ("median_ms", "p95_ms"):
            gates[f"python-{metric}"] = float(report[metric]) < python[metric]
        gates["python-ready"] = float(report["model_load_ms"]) < python["ready_from_python_entry_ms"]
    result["cases"].append({"id": case["id"], "byte_identical_pcm": True, "report": report, "speed_gates": gates})
    (args.output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert report["correctness_passed"] and all(gates.values()), (case["id"], gates)
    print(f"{case['id']}: identical WAV; load={report['model_load_ms']} median={report['median_ms']} p95={report['p95_ms']} ms; all speed gates pass", flush=True)
