"""Exercise native acquisition from an empty isolated cache, then without a server."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("receipts", type=Path)
    args = parser.parse_args()
    root = args.receipts.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "home").exists() or (root / "cache").exists():
        raise RuntimeError("use an empty receipt directory to prove first acquisition")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(args.archive.stat().st_size))
            self.end_headers()
            with args.archive.open("rb") as archive:
                shutil.copyfileobj(archive, self.wfile)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(os.environ)
    for name in ["TEAMY_TTS_NATIVE_MODEL_DIR", "TEAMY_TTS_MODEL_DIR", "TEAMY_TTS_BACKEND"]:
        env.pop(name, None)
    env.update(TEAMY_TTS_HOME_DIR=str(root / "home"), TEAMY_TTS_CACHE_DIR=str(root / "cache"),
               TEAMY_TTS_TEAMY_CUDA_SOURCE_URL=f"http://127.0.0.1:{server.server_port}/model.zip")

    def run(name, *command, success=True, parse_json=True):
        result = subprocess.run([str(args.executable), "--output-format", "json", *command],
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        (root / f"{name}.stdout").write_text(result.stdout, encoding="utf-8")
        (root / f"{name}.stderr").write_text(result.stderr, encoding="utf-8")
        if (result.returncode == 0) != success:
            raise AssertionError(f"{name}: exit {result.returncode}: {result.stderr}")
        return json.loads(result.stdout) if success and parse_json else None

    try:
        acquired = run("fresh", "model", "acquire-prepared", "Teamy")
        assert acquired["verified"] and acquired["configured"]
        prepared = Path(acquired["prepared_dir"])
        assert {p.name for p in prepared.iterdir()} == {"weights.safetensors", "frontend.tsv", "manifest.json"}
        with (prepared / "weights.safetensors").open("rb") as f:
            weights_hash = hashlib.file_digest(f, "sha256").hexdigest()
        assert weights_hash == "f3ab504f0fb9f360c05d8ffd044cdea3f1ca00666fd51f8762031e3baf77b87b"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    again = run("offline", "model", "acquire-prepared", "Teamy")
    assert again["verified"] and not again["configured"]
    assert requests == ["/model.zip"]
    doctor = run("doctor", "doctor", "--offline", "--deep")
    assert doctor["status"] == "pass", doctor
    run("speech", "write", "Hello, friend", "--output", str(root / "native.wav"), parse_json=False)
    bad = root / "bad.zip"
    bad.write_bytes(b"not a model archive")
    run("corrupt", "model", "prepare", "glados", "--source-archive", str(bad), success=False)
    # Archive pinning must reject a valid ZIP with a traversal entry before extraction.
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../escaped.txt", "must not be written")
    run("traversal", "model", "prepare", "glados", "--source-archive", str(bad), success=False)
    assert not (root / "escaped.txt").exists()
    # A previously installed model is reverified on offline acquisition.
    frontend = prepared / "frontend.tsv"
    original = frontend.read_bytes()
    try:
        frontend.write_bytes(b"!" + original[1:])
        run("changed-file", "model", "acquire-prepared", "Teamy", success=False)
        frontend.unlink()
        run("missing-file", "model", "acquire-prepared", "Teamy", success=False)
    finally:
        frontend.write_bytes(original)
    assert not list((root / "cache").rglob(".teamy-staging-*")), "partial download/extraction left behind"
    report = {"empty_cache": True, "single_download": True, "offline_reuse": True,
              "deep_doctor": True, "native_synthesis": True, "corrupt_rejected": True,
              "traversal_rejected": True, "changed_file_rejected": True, "missing_file_rejected": True,
              "weights_sha256": weights_hash}
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
