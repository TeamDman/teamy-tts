"""Check the real DLL's COM ownership and format ABI without registration or audio."""
import ctypes as c
import json
from pathlib import Path
import sys
import uuid


class Guid(c.Structure):
    _fields_ = [("a", c.c_uint32), ("b", c.c_uint16), ("c", c.c_uint16), ("d", c.c_ubyte * 8)]

    @classmethod
    def parse(cls, text):
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


class Wave(c.Structure):
    _pack_ = 1
    _fields_ = [("tag", c.c_uint16), ("channels", c.c_uint16), ("rate", c.c_uint32),
                ("bytes_sec", c.c_uint32), ("align", c.c_uint16), ("bits", c.c_uint16), ("extra", c.c_uint16)]


def method(obj, slot, result, *args):
    table = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(result, c.c_void_p, *args)(table[slot])


def release(obj):
    return method(obj, 2, c.c_uint32)(obj)


def main():
    dll = c.WinDLL(str(Path(sys.argv[1]).resolve()))
    modules = (c.c_void_p * 1024)()
    needed = c.c_uint32()
    process = c.windll.kernel32.GetCurrentProcess
    process.restype = c.c_void_p
    enum = c.windll.psapi.EnumProcessModules
    enum.argtypes = [c.c_void_p, c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32)]
    assert enum(process(), modules, c.sizeof(modules), c.byref(needed))
    names = []
    module_name = c.windll.kernel32.GetModuleFileNameW
    module_name.argtypes = [c.c_void_p, c.c_wchar_p, c.c_uint32]
    for module in modules[:needed.value // c.sizeof(c.c_void_p)]:
        name = c.create_unicode_buffer(32768)
        assert module_name(module, name, len(name))
        names.append(Path(name.value).name.lower())
    assert not any(name.startswith(("torch", "c10", "cud", "cublas", "nvrtc")) for name in names), names
    dll.DllCanUnloadNow.restype = c.c_long
    dll.DllGetClassObject.argtypes = [c.POINTER(Guid), c.POINTER(Guid), c.POINTER(c.c_void_p)]
    dll.DllGetClassObject.restype = c.c_long
    clsid = Guid.parse("3d190e91-1f23-4bf5-a136-36620c2e406e")
    iid_factory = Guid.parse("00000001-0000-0000-c000-000000000046")
    iid_engine = Guid.parse("a74d7c8e-4cc5-4f2f-a6eb-804dee18500e")
    assert dll.DllCanUnloadNow() == 0
    factory = c.c_void_p()
    assert dll.DllGetClassObject(c.byref(clsid), c.byref(iid_factory), c.byref(factory)) == 0
    assert dll.DllCanUnloadNow() == 1
    create = method(factory, 3, c.c_long, c.c_void_p, c.POINTER(Guid), c.POINTER(c.c_void_p))
    engine = c.c_void_p()
    assert create(factory, None, c.byref(iid_engine), c.byref(engine)) == 0
    assert release(factory) == 0
    assert dll.DllCanUnloadNow() == 1
    get_format = method(engine, 4, c.c_long, c.c_void_p, c.c_void_p, c.POINTER(Guid), c.POINTER(c.c_void_p))
    format_id = Guid()
    allocation = c.c_void_p()
    assert get_format(engine, None, None, c.byref(format_id), c.byref(allocation)) == 0
    wave = c.cast(allocation, c.POINTER(Wave)).contents
    assert (wave.tag, wave.channels, wave.rate, wave.bytes_sec, wave.align, wave.bits, wave.extra) == (1, 1, 22050, 44100, 2, 16, 0)
    assert bytes(format_id) == uuid.UUID("c31adbae-527f-4ff5-a230-f62bb61ff70c").bytes_le
    c.windll.ole32.CoTaskMemFree.argtypes = [c.c_void_p]
    c.windll.ole32.CoTaskMemFree(allocation)
    assert get_format(engine, None, None, None, c.byref(allocation)) == -2147467261  # E_POINTER
    assert release(engine) == 0
    assert dll.DllCanUnloadNow() == 0
    print(json.dumps({"factory_lifetime": True, "engine_lifetime": True, "pcm_format": True, "null_output_guard": True, "unloadable": True, "no_gpu_or_torch_modules": True}))


if __name__ == "__main__":
    main()
