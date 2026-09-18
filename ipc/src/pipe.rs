//! Local-only nonblocking named pipes. Every handle stays on its owning thread.
use crate::protocol::{self, Frame, MAX_BUFFER};
use std::{collections::VecDeque, io, ptr};
use windows::{
    Win32::{
        Foundation::*,
        Security::{Authorization::*, *},
        Storage::FileSystem::*,
        System::{Pipes::*, RemoteDesktop::*, Threading::*},
    },
    core::{Owned, PCWSTR, PWSTR},
};

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}
fn win(e: windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(e.code().0 & 0xffff)
}
struct LocalAllocation(HLOCAL);
impl Drop for LocalAllocation {
    fn drop(&mut self) {
        unsafe {
            let _ = LocalFree(Some(self.0));
        }
    }
}

pub struct Endpoint {
    pub name: String,
    sid: String,
}
impl Endpoint {
    pub fn current(instance: &str) -> io::Result<Self> {
        if instance.is_empty()
            || instance.len() > 64
            || !instance
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-')
        {
            return Err(protocol::invalid("invalid worker instance name"));
        }
        let sid = process_sid(unsafe { GetCurrentProcess() })?;
        let mut session = 0;
        unsafe {
            ProcessIdToSessionId(GetCurrentProcessId(), &mut session).map_err(win)?;
        }
        Ok(Self {
            name: format!(r"\\.\pipe\teamy-tts-v1-{sid}-{session}-{instance}"),
            sid,
        })
    }
}

fn process_sid(process: HANDLE) -> io::Result<String> {
    let mut handle = HANDLE::default();
    unsafe {
        OpenProcessToken(process, TOKEN_QUERY, &mut handle).map_err(win)?;
    }
    let handle = unsafe { Owned::new(handle) };
    let mut len = 0;
    unsafe {
        let _ = GetTokenInformation(*handle, TokenUser, None, 0, &mut len);
    }
    // u64 backing storage preserves the alignment required by TOKEN_USER.
    let mut data = vec![0u64; (len as usize).div_ceil(8)];
    unsafe {
        GetTokenInformation(
            *handle,
            TokenUser,
            Some(data.as_mut_ptr().cast()),
            len,
            &mut len,
        )
        .map_err(win)?;
    }
    let user = unsafe { &*data.as_ptr().cast::<TOKEN_USER>() };
    let mut sid = PWSTR::null();
    unsafe {
        ConvertSidToStringSidW(user.User.Sid, &mut sid).map_err(win)?;
    }
    let _allocation = LocalAllocation(HLOCAL(sid.0.cast()));
    unsafe { sid.to_string() }.map_err(|_| protocol::invalid("invalid user SID"))
}

/// A machine-wide COM engine can also load in elevated applications. Never
/// launch a user-installed worker with the caller's elevated token.
pub fn is_elevated() -> io::Result<bool> {
    let mut token = HANDLE::default();
    unsafe {
        OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).map_err(win)?;
    }
    let token = unsafe { Owned::new(token) };
    let mut elevation = TOKEN_ELEVATION::default();
    let mut size = 0;
    unsafe {
        GetTokenInformation(
            *token,
            TokenElevation,
            Some((&mut elevation as *mut TOKEN_ELEVATION).cast()),
            size_of::<TOKEN_ELEVATION>() as u32,
            &mut size,
        )
        .map_err(win)?;
    }
    Ok(elevation.TokenIsElevated != 0)
}

pub struct Pipe {
    handle: Owned<HANDLE>,
    input: Vec<u8>,
    output: VecDeque<Vec<u8>>,
    sent: usize,
    queued: usize,
}
impl Pipe {
    pub fn server_process(&self) -> io::Result<Owned<HANDLE>> {
        let mut pid = 0;
        unsafe {
            GetNamedPipeServerProcessId(*self.handle, &mut pid).map_err(win)?;
        }
        unsafe {
            Ok(Owned::new(
                OpenProcess(PROCESS_SYNCHRONIZE, false, pid).map_err(win)?,
            ))
        }
    }
    fn new(handle: Owned<HANDLE>) -> Self {
        Self {
            handle,
            input: Vec::new(),
            output: VecDeque::new(),
            sent: 0,
            queued: 0,
        }
    }
    pub fn listen(endpoint: &Endpoint, first: bool) -> io::Result<Self> {
        let name = wide(&endpoint.name);
        let sddl = wide(&format!("D:P(A;;GA;;;{})", endpoint.sid));
        let mut descriptor = PSECURITY_DESCRIPTOR(ptr::null_mut());
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                PCWSTR(sddl.as_ptr()),
                1,
                &mut descriptor,
                None,
            )
            .map_err(win)?;
        }
        let _descriptor = LocalAllocation(HLOCAL(descriptor.0));
        let security = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: descriptor.0,
            bInheritHandle: false.into(),
        };
        let flags = PIPE_ACCESS_DUPLEX
            | if first {
                FILE_FLAG_FIRST_PIPE_INSTANCE
            } else {
                FILE_FLAGS_AND_ATTRIBUTES(0)
            };
        let handle = unsafe {
            CreateNamedPipeW(
                PCWSTR(name.as_ptr()),
                flags,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
                17,
                65536,
                65536,
                0,
                Some(&security),
            )
        };
        if handle.is_invalid() {
            return Err(win(windows::core::Error::from_thread()));
        }
        Ok(Self::new(unsafe { Owned::new(handle) }))
    }
    pub fn accept(&self) -> io::Result<bool> {
        match unsafe { ConnectNamedPipe(*self.handle, None) } {
            Ok(()) => Ok(true),
            Err(e) if e.code() == ERROR_PIPE_CONNECTED.to_hresult() => Ok(true),
            Err(e) if e.code() == ERROR_PIPE_LISTENING.to_hresult() => Ok(false),
            Err(e) => Err(win(e)),
        }
    }
    pub fn connect(endpoint: &Endpoint) -> io::Result<Self> {
        let name = wide(&endpoint.name);
        let handle = unsafe {
            Owned::new(
                CreateFileW(
                    PCWSTR(name.as_ptr()),
                    GENERIC_READ.0 | GENERIC_WRITE.0,
                    FILE_SHARE_MODE(0),
                    None,
                    OPEN_EXISTING,
                    SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION,
                    None,
                )
                .map_err(win)?,
            )
        };
        // Check the server identity before sending narration. A predictable pipe
        // name alone must not trust a process belonging to another Windows user.
        let mut pid = 0;
        unsafe {
            GetNamedPipeServerProcessId(*handle, &mut pid).map_err(win)?;
        }
        let process = unsafe {
            Owned::new(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).map_err(win)?)
        };
        if process_sid(*process)? != endpoint.sid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "worker belongs to another Windows user",
            ));
        }
        unsafe {
            SetNamedPipeHandleState(
                *handle,
                Some(&(PIPE_READMODE_BYTE | PIPE_NOWAIT)),
                None,
                None,
            )
            .map_err(win)?;
        }
        Ok(Self::new(handle))
    }
    pub fn enqueue(&mut self, frame: Frame) -> io::Result<()> {
        let bytes = frame.encode()?;
        if self.queued + bytes.len() > MAX_BUFFER {
            return Err(protocol::invalid("worker output queue full"));
        }
        self.queued += bytes.len();
        self.output.push_back(bytes);
        Ok(())
    }
    pub fn discard_output(&mut self) {
        // Never truncate an already-started frame: preserve framing, cancel by id at the receiver.
        if self.sent == 0 {
            self.output.clear();
            self.queued = 0;
        } else {
            self.output.truncate(1);
            self.queued = self.output.front().map_or(0, |b| b.len() - self.sent);
        }
    }
    pub fn pump(&mut self) -> io::Result<Vec<Frame>> {
        for _ in 0..8 {
            let Some(bytes) = self.output.front() else {
                break;
            };
            let mut count = 0;
            match unsafe {
                WriteFile(
                    *self.handle,
                    Some(&bytes[self.sent..]),
                    Some(&mut count),
                    None,
                )
            } {
                Ok(()) => {}
                Err(e) if e.code() == ERROR_NO_DATA.to_hresult() => break,
                Err(e) => return Err(win(e)),
            }
            if count == 0 {
                break;
            }
            self.sent += count as usize;
            self.queued -= count as usize;
            if self.sent == bytes.len() {
                self.output.pop_front();
                self.sent = 0;
            }
        }
        let mut frames = Vec::new();
        let mut bytes = [0u8; 8192];
        for _ in 0..16 {
            let mut count = 0;
            match unsafe { ReadFile(*self.handle, Some(&mut bytes), Some(&mut count), None) } {
                Ok(()) => {}
                Err(e) if e.code() == ERROR_NO_DATA.to_hresult() => break,
                Err(e) => return Err(win(e)),
            }
            if count == 0 {
                break;
            }
            if self.input.len() + count as usize > MAX_BUFFER {
                return Err(protocol::invalid("worker input queue full"));
            }
            self.input.extend_from_slice(&bytes[..count as usize]);
            while let Some(frame) = protocol::decode(&mut self.input)? {
                frames.push(frame);
                if frames.len() >= 128 {
                    return Err(protocol::invalid("too many frames"));
                }
            }
        }
        Ok(frames)
    }
}
