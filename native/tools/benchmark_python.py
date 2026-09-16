"""Synchronized Python reference timings, using the real upstream API or vocoder.

Run each invocation in a fresh process. No playback, downloads or model edits.
The actual upstream path preserves its frontend and WAV encoding overhead.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from time import perf_counter

ENTRY = perf_counter()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["upstream", "vocoder"], required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--text", default="Hello, friend")
    parser.add_argument("--voice", choices=["p1", "p2"], default="p2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lifecycle-markers", action="store_true", help="Emit flushed external launch timing markers")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    output = args.output.resolve()
    fixture = args.fixture.resolve() if args.fixture else None
    upstream = args.upstream_root.resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    imported = perf_counter()
    import torch
    import numpy as np
    import_ms = (perf_counter() - imported)*1000
    load_start = perf_counter()
    if args.mode == "upstream":
        from glados import tts_runner
        model = tts_runner(use_p1=args.voice == "p1", log=False)

        def run():
            audio = model.run_tts(args.text, args.alpha)
            return np.frombuffer(audio.raw_data, dtype=np.int16).copy()
        boundary = "upstream run_tts: text to complete int16 PCM, including WAV encode/decode"
    else:
        if fixture is None:
            parser.error("vocoder requires --fixture")
        data = fixture.read_bytes()
        header_size = struct.unpack_from("<Q", data)[0]
        header = json.loads(data[8:8+header_size])
        desc = header["mel"]
        assert desc["dtype"] == "F32"
        lo, hi = desc["data_offsets"]
        mel = torch.from_numpy(np.frombuffer(data[8+header_size+lo:8+header_size+hi], dtype="<f4").copy()).reshape(desc["shape"])
        model = torch.jit.load("models/vocoder-gpu.pt", map_location="cuda:0").eval()

        def run():
            with torch.inference_mode():
                return model(mel.cuda()).cpu().contiguous().numpy().copy()
        boundary = "host mel to complete host F32 PCM, including transfers and allocation"
    torch.cuda.synchronize()
    load_ms = (perf_counter() - load_start)*1000
    ready_ms = (perf_counter() - ENTRY)*1000
    if args.lifecycle_markers:
        print("GLADOS_READY", flush=True)
    first_start = perf_counter()
    audio = run()
    first_ms = (perf_counter() - first_start)*1000
    if args.lifecycle_markers:
        print("GLADOS_FIRST_PCM", flush=True)
    expected_samples = audio.size
    assert expected_samples > 0 and np.isfinite(audio).all()
    for _ in range(3):
        run()
    timings = []
    for _ in range(args.iterations):
        start = perf_counter()
        audio = run()
        timings.append((perf_counter() - start)*1000)
        assert audio.size == expected_samples and np.isfinite(audio).all()
    ordered = sorted(timings)
    result = {"backend": "upstream-python", "stage": args.mode, "torch": torch.__version__,
              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "torch_threads": torch.get_num_threads(), "text": args.text,
              "voice": args.voice, "alpha": args.alpha, "import_ms": import_ms,
              "load_ms": load_ms, "ready_from_python_entry_ms": ready_ms,
              "first_ms": first_ms, "first_pcm_from_python_entry_ms": ready_ms+first_ms,
              "measurement_ms": timings, "median_ms": ordered[len(ordered)//2],
              "p95_ms": ordered[math.ceil(len(ordered)*0.95)-1], "sample_count": expected_samples,
              "timing_boundary": boundary,
              "correctness_gate": "finite and stable sample count; waveform equivalence is a separate gate"}
    if os.name == "nt":
        import ctypes as c
        from ctypes import wintypes as w
        class Memory(c.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [(k, c.c_size_t) for k in
                ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                 "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage"]]
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = w.HANDLE
        psapi = c.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Memory), w.DWORD]
        psapi.GetProcessMemoryInfo.restype = w.BOOL
        memory = Memory()
        memory.cb = c.sizeof(memory)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), c.byref(memory), memory.cb):
            raise c.WinError(c.get_last_error())
        result["host_memory"] = {"working_set_bytes": memory.WorkingSetSize,
            "peak_working_set_bytes": memory.PeakWorkingSetSize, "private_bytes": memory.PrivateUsage}
    result["torch_peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    result["torch_peak_reserved_gpu_bytes"] = torch.cuda.max_memory_reserved()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
