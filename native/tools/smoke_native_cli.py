"""Check native CLI configuration, diagnostics and cancellation in an isolated home."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
binary = args.binary.resolve()
models = args.models.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
for key in ("TEAMY_TTS_NATIVE_MODEL_DIR", "TEAMY_TTS_BACKEND", "TEAMY_TTS_TORCH_MODEL_DIR",
            "TEAMY_TTS_MODEL_DIR", "TEAMY_TTS_TORCH_DEVICE"):
    env.pop(key, None)
env["TEAMY_TTS_HOME_DIR"] = str(output / "home")
env["NO_COLOR"] = "1"

def run(name, arguments, overrides=None):
    result = subprocess.run([str(binary), "--output-format", "json", *arguments],
        env=env | (overrides or {}), capture_output=True, encoding="utf-8", timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW)
    (output / f"{name}.stdout.json").write_text(result.stdout, encoding="utf-8")
    (output / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name}: {result.returncode}: {result.stderr[-1000:]}")
    return json.loads(result.stdout)

configured = run("set", ["config", "set", "--backend", "cuda-native", "--native-model-dir", str(models)])
assert Path(configured["effective"]["native_model_dir"]) == models
shown = run("show", ["config", "show"])
assert shown["effective"]["backend"] == "cuda-native"
override = run("override", ["config", "show"], {"TEAMY_TTS_NATIVE_MODEL_DIR": str(output / "override")})
assert Path(override["effective"]["native_model_dir"]) == output / "override"
doctor = run("doctor", ["doctor", "--offline", "--deep"])
assert doctor["status"] == "pass", doctor

# Keep stdin open and empty: EOF must not be the reason this process exits.
started = time.perf_counter()
with (output / "cancel.stdout.txt").open("w", encoding="utf-8") as stdout, \
     (output / "cancel.stderr.txt").open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen([str(binary), "--stop-after-duration", "2s", "interactive", "--volume", "0"],
        stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        process.wait(timeout=15)
    finally:
        process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
elapsed = (time.perf_counter()-started)*1000
log = (output / "cancel.stderr.txt").read_text(encoding="utf-8") + (output / "cancel.stdout.txt").read_text(encoding="utf-8")
assert "interactive mode ready" in log and "cancelled while waiting for stdin" in log, log[-2000:]
assert 1800 <= elapsed < 10000, elapsed
cleared = run("clear", ["config", "clear", "--native-model-dir", "--backend"])
assert cleared["stored"]["native_model_dir"] is None
assert cleared["effective"]["native_model_dir"] is None
assert not list((output / "home").rglob("*.wav"))
receipt = {"config_set_show_override_clear": True, "deep_doctor": True,
           "cancellation_while_stdin_open": True, "cancel_exit_code": process.returncode,
           "cancel_elapsed_ms": elapsed, "no_implicit_wav_files": True}
(output / "checks.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
