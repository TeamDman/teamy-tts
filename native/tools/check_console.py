"""Exercise interactive stdin using a private, hidden Windows console.

No keystrokes are sent to the user's terminal. The worker owns its console and
uses WriteConsoleInputW; piping an EOT byte cannot test Windows cooked input.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--worker',action='store_true')
parser.add_argument('--expect-broken',action='store_true')
args=parser.parse_args()
if os.name != 'nt':
    parser.error('This check requires a Windows console.')
args.output.mkdir(parents=True,exist_ok=True)

if not args.worker:
    startup=subprocess.STARTUPINFO()
    startup.dwFlags=subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=0
    run=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
        creationflags=subprocess.CREATE_NEW_CONSOLE,startupinfo=startup,
        capture_output=True,text=True,encoding='utf-8',timeout=180)
    print(run.stdout,end='')
    print(run.stderr,end='',file=sys.stderr)
    sys.exit(run.returncode)

kernel=c.WinDLL('kernel32',use_last_error=True)
class Char(c.Union):
    _fields_=[('UnicodeChar',w.WCHAR),('AsciiChar',c.c_char)]
class Key(c.Structure):
    _fields_=[('bKeyDown',w.BOOL),('wRepeatCount',w.WORD),('wVirtualKeyCode',w.WORD),
              ('wVirtualScanCode',w.WORD),('uChar',Char),('dwControlKeyState',w.DWORD)]
class Event(c.Union):
    _fields_=[('KeyEvent',Key),('padding',c.c_byte*16)]
class Input(c.Structure):
    _fields_=[('EventType',w.WORD),('Event',Event)]
kernel.WriteConsoleInputW.argtypes=[w.HANDLE,c.POINTER(Input),w.DWORD,c.POINTER(w.DWORD)]
kernel.WriteConsoleInputW.restype=w.BOOL
kernel.FlushConsoleInputBuffer.argtypes=[w.HANDLE]
kernel.FlushConsoleInputBuffer.restype=w.BOOL
import msvcrt
console_in=open('CONIN$','rb',buffering=0)
console_out=open('CONOUT$','wb',buffering=0)
handle=msvcrt.get_osfhandle(console_in.fileno())
def type_text(text):
    events=[]
    # UTF-16 code units also exercise surrogate pairs across the console API.
    encoded=text.encode('utf-16-le')
    for i in range(0,len(encoded),2):
        char=chr(int.from_bytes(encoded[i:i+2],'little'))
        for down in [True,False]:
            event=Input()
            event.EventType=1
            key=event.Event.KeyEvent
            key.bKeyDown=down
            key.wRepeatCount=1
            key.wVirtualKeyCode={'\x04':0x44,'\x1a':0x5a,'\r':0x0d,'\b':0x08}.get(char,0)
            key.uChar.UnicodeChar=char
            key.dwControlKeyState=8 if char in ['\x04','\x1a'] else 0
            events.append(event)
    buffer=(Input*len(events))(*events)
    written=w.DWORD()
    if not kernel.WriteConsoleInputW(handle,buffer,len(events),c.byref(written)):
        raise c.WinError(c.get_last_error())
    assert written.value==len(events)

receipts=[]
cases=[('empty',[],False)] if args.expect_broken else [
    ('empty',[],False),
    ('line',[('Hello.\r',1)],False),
    ('partial',[('Hello.\x04',1)],False),
    ('unicode-backspace',[('Caféx\b.\r',1)],False),
    ('redirected-stdout',[],True),
    ('ctrl-z',[],False),
]
for name,steps,redirect in cases:
    assert kernel.FlushConsoleInputBuffer(handle)
    env=os.environ.copy()
    env['NO_COLOR']='1'
    process=subprocess.Popen([str(args.binary.resolve()),'interactive','--volume','0'],
        stdin=console_in,stdout=subprocess.PIPE if redirect else console_out,
        stderr=subprocess.PIPE,env=env)
    messages=queue.Queue()
    lines=[]
    def consume():
        for line in iter(process.stderr.readline,b''):
            decoded=line.decode('utf-8',errors='replace')
            lines.append(decoded)
            messages.put(decoded)
    reader=threading.Thread(target=consume,daemon=True)
    reader.start()
    def wait_for(marker,seconds=30):
        deadline=time.monotonic()+seconds
        while True:
            try:
                line=messages.get(timeout=max(0.01,deadline-time.monotonic()))
            except queue.Empty:
                raise AssertionError(f'{name}: missing {marker}; log={lines[-10:]}')
            if marker in line:
                return
    try:
        wait_for('interactive mode ready')
        for text,count in steps:
            type_text(text)
            for _ in range(count):
                wait_for('interactive playback complete')
        started=time.perf_counter()
        type_text('\x1a\r' if name=='ctrl-z' else '\x04')
        try:
            code=process.wait(timeout=2 if args.expect_broken else 5)
            exited=True
        except subprocess.TimeoutExpired:
            exited=False
            code=None
        elapsed=(time.perf_counter()-started)*1000
        reader.join(timeout=0.1)
        receipt=dict(case=name,exit_code=code,exited=exited,exit_ms=elapsed,
            syntheses=sum('interactive playback complete' in line for line in lines))
        receipts.append(receipt)
        print(json.dumps(receipt),flush=True)
        if args.expect_broken:
            assert not exited, 'baseline unexpectedly already handles Ctrl-D'
        else:
            assert exited and code==0, receipt
            assert 'interactive mode finished' in ''.join(lines), lines[-10:]
            assert receipt['syntheses']==sum(count for _,count in steps), receipt
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=2)
        (args.output/f'{name}.log').write_text(''.join(lines),encoding='utf-8')
        (args.output/'checks.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')

if not args.expect_broken:
    for name,payload,count in [('pipe-empty',b'',0),('pipe-line',b'Hello.\n',1)]:
        run=subprocess.run([str(args.binary.resolve()),'interactive','--volume','0'],
            input=payload,capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        log=run.stderr.decode('utf-8',errors='replace')
        assert run.returncode==0 and 'interactive mode finished' in log,log
        assert log.count('interactive playback complete')==count,log
        receipts.append(dict(case=name,exit_code=run.returncode,syntheses=count))
    failed=subprocess.run([str(args.binary.resolve()),'interactive','--volume','0'],
        input=b'\xff\n',capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    assert failed.returncode != 0, 'Invalid UTF-8 must report a read error, not clean EOF'
    receipts.append(dict(case='pipe-read-error',exit_code=failed.returncode))
    (args.output/'checks.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print('All console and pipe cases passed.',flush=True)
