"""Measure actual upstream Python launch to readiness and first complete host PCM.

Runs benchmark_python.py in a separate process; no model or API substitutions.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--python", type=Path, required=True)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--text", default="Hello, friend")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
child_output = args.output.with_name(args.output.stem + "-child.json")
started = time.perf_counter()
process = subprocess.Popen([str(args.python), str(Path(__file__).with_name("benchmark_python.py")),
    "--mode", "upstream", "--upstream-root", str(args.upstream), "--text", args.text,
    "--iterations", "20", "--lifecycle-markers", "--output", str(child_output.resolve())],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
    creationflags=subprocess.CREATE_NO_WINDOW)
# Drain both streams so warnings cannot fill a pipe while a model is loading.
import queue
import threading
events = queue.Queue()
def read(stream, name):
    for line in stream:
        events.put((time.perf_counter(), name, line.rstrip()))
    events.put((time.perf_counter(), name, None))
threads = [threading.Thread(target=read, args=(stream, name), daemon=True)
           for stream, name in [(process.stdout, "stdout"), (process.stderr, "stderr")]]
for thread in threads:
    thread.start()
try:
    markers = {}
    log = []
    ended = 0
    deadline = time.monotonic() + 180
    while ended < 2:
        timestamp, stream, line = events.get(timeout=max(0.01, deadline-time.monotonic()))
        if line is None:
            ended += 1
        else:
            log.append({"ms": (timestamp-started)*1000, "stream": stream, "line": line})
            if line in ("GLADOS_READY", "GLADOS_FIRST_PCM"):
                markers[line] = (timestamp-started)*1000
    process.wait(timeout=15)
    if process.returncode or len(markers) != 2:
        raise RuntimeError(f"Python lifecycle failed: {process.returncode}, {log[-10:]}")
    result = {"launch_to_ready_ms": markers["GLADOS_READY"],
              "launch_to_first_pcm_ms": markers["GLADOS_FIRST_PCM"],
              "exit_code": process.returncode,
              "child_report": json.loads(child_output.read_text(encoding="utf-8")), "log": log}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("log", "child_report")}))
finally:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=15)
