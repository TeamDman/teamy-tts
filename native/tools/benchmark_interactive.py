"""External Windows interactive lifecycle probe; playback uses zero volume.

Reads the existing product's log events, without instrumenting its binary.
The handoff metric ends immediately before the synchronous audio API call;
it does not claim to measure physical speaker latency.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--text", action="append")
parser.add_argument("--queue-first", action="store_true", help="Queue the first request during startup")
args = parser.parse_args()
texts = args.text or ["Hello, friend", "Quux.", "Hello, friend. Hello again!"]
events = queue.Queue()
started = time.perf_counter()
env = os.environ.copy()
env["NO_COLOR"] = "1"
process = subprocess.Popen([str(args.binary.resolve()), "interactive", "--volume", "0"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
    env=env, creationflags=subprocess.CREATE_NO_WINDOW, bufsize=1)

def read(stream, name):
    for line in stream:
        events.put((time.perf_counter(), name, line.rstrip()))
    events.put((time.perf_counter(), name, None))

threads = [threading.Thread(target=read, args=(stream, name), daemon=True)
           for stream, name in [(process.stdout, "stdout"), (process.stderr, "stderr")]]
for thread in threads:
    thread.start()
first_sent = None
if args.queue_first:
    first_sent = time.perf_counter()
    process.stdin.write(texts[0] + "\n")
    process.stdin.flush()
log = []

def until(marker):
    deadline = time.monotonic() + 120
    while True:
        timestamp, stream, line = events.get(timeout=max(0.01, deadline - time.monotonic()))
        if line is None:
            raise RuntimeError(f"child ended before {marker!r}; exit={process.poll()}, log={log[-8:]}")
        log.append({"ms": (timestamp-started)*1000, "stream": stream, "line": line})
        if marker in line:
            return timestamp

def snapshot():
    psapi = c.WinDLL("psapi", use_last_error=True)
    psapi.EnumProcessModulesEx.argtypes = [w.HANDLE, c.POINTER(w.HMODULE), w.DWORD, c.POINTER(w.DWORD), w.DWORD]
    psapi.EnumProcessModulesEx.restype = w.BOOL
    psapi.GetModuleFileNameExW.argtypes = [w.HANDLE, w.HMODULE, w.LPWSTR, w.DWORD]
    psapi.GetModuleFileNameExW.restype = w.DWORD
    class Memory(c.Structure):
        _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [(k, c.c_size_t) for k in
            ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage"]]
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Memory), w.DWORD]
    psapi.GetProcessMemoryInfo.restype = w.BOOL
    modules = (w.HMODULE * 2048)()
    required = w.DWORD()
    handle = w.HANDLE(int(process._handle))
    if not psapi.EnumProcessModulesEx(handle, modules, c.sizeof(modules), c.byref(required), 3):
        raise c.WinError(c.get_last_error())
    if required.value > c.sizeof(modules):
        raise RuntimeError("module buffer too small")
    names = []
    for module in modules[:required.value // c.sizeof(w.HMODULE)]:
        name = c.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(handle, module, name, len(name)):
            raise c.WinError(c.get_last_error())
        names.append(Path(name.value).name)
    memory = Memory()
    memory.cb = c.sizeof(memory)
    if not psapi.GetProcessMemoryInfo(handle, c.byref(memory), memory.cb):
        raise c.WinError(c.get_last_error())
    return {"loaded_module_names": sorted(names), "working_set_bytes": memory.WorkingSetSize,
            "peak_working_set_bytes": memory.PeakWorkingSetSize, "private_bytes": memory.PrivateUsage,
            "loads_torch": any(name.lower().startswith(("torch", "c10")) and name.lower().endswith(".dll") for name in names)}

try:
    ready = until("interactive mode ready")
    result = {"launch_to_ready_ms": (ready-started)*1000, "requests": [], "first_request_queued_during_startup": args.queue_first,
              "boundary": "External process launch/readiness and stdin flush to host WAV/audio API handoff. Zero-volume real synchronous playback; hardware latency unmeasured."}
    for index, text in enumerate(texts):
        if index == 0 and first_sent is not None:
            sent = first_sent
        else:
            sent = time.perf_counter()
            process.stdin.write(text + "\n")
            process.stdin.flush()
        handoff = until("playing interactive in-memory WAV output")
        complete = until("interactive playback complete")
        result["requests"].append({"text": text, "stdin_to_audio_handoff_ms": (handoff-sent)*1000,
                                   "launch_to_audio_handoff_ms": (handoff-started)*1000,
                                   "playback_call_ms": (complete-handoff)*1000})
    result["final_resources"] = snapshot()
    result["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    process.stdin.close()
    process.wait(timeout=30)
    for thread in threads:
        thread.join(timeout=5)
    if process.returncode:
        raise RuntimeError(f"interactive exit {process.returncode}")
    result["exit_code"] = process.returncode
    result["log"] = log
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("log", "ready_resources", "final_resources")}, indent=2))
finally:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=15)
