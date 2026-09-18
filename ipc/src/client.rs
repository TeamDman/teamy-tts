use crate::{
    pipe::{Endpoint, Pipe},
    protocol::*,
};
use std::{
    io,
    os::windows::process::CommandExt,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

pub struct Client {
    pipe: Pipe,
    next_id: u64,
    pub pid: u32,
}

/// Inspect a running worker without starting one.
pub fn status(instance: &str) -> io::Result<Status> {
    let mut pipe = Pipe::connect(&Endpoint::current(instance)?)?;
    pipe.enqueue(Frame::empty(HELLO, 0))?;
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        for frame in pipe.pump()? {
            if frame.kind == STATUS {
                return frame.parse();
            }
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "worker status timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

pub fn stop(instance: &str) -> io::Result<()> {
    use windows::Win32::{
        Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT},
        System::Threading::WaitForSingleObject,
    };
    let endpoint = Endpoint::current(instance)?;
    let mut pipe = match Pipe::connect(&endpoint) {
        Ok(pipe) => pipe,
        Err(e) if e.raw_os_error() == Some(2) => return Ok(()),
        Err(e) => return Err(e),
    };
    let process = pipe.server_process()?;
    pipe.enqueue(Frame::empty(HELLO, 0))?;
    pipe.enqueue(Frame::empty(STOP, 0))?;
    pipe.pump()?;
    drop(pipe);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match unsafe { WaitForSingleObject(*process, 0) } {
            WAIT_OBJECT_0 => return Ok(()),
            WAIT_TIMEOUT => {}
            _ => return Err(io::Error::last_os_error()),
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "worker shutdown timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
impl Client {
    /// `tick` is called on the caller's thread, so a COM engine can poll its site.
    /// Returning false cancels startup; no request is replayed after acceptance.
    pub fn start(
        worker: &Path,
        instance: &str,
        mut tick: impl FnMut() -> bool,
    ) -> io::Result<Self> {
        let endpoint = Endpoint::current(instance)?;
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut launched = false;
        let mut pipe = loop {
            if !tick() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "speech cancelled",
                ));
            }
            match Pipe::connect(&endpoint) {
                Ok(p) => break p,
                Err(e) if matches!(e.raw_os_error(), Some(2 | 231)) => {
                    if !launched && e.raw_os_error() == Some(2) {
                        if crate::pipe::is_elevated()? {
                            return Err(io::Error::new(
                                io::ErrorKind::PermissionDenied,
                                "start Teamy TTS from a normal, non-administrator session before using speech in an elevated application",
                            ));
                        }
                        if !worker.is_absolute() || !worker.is_file() {
                            return Err(io::Error::new(
                                io::ErrorKind::NotFound,
                                "registered Teamy TTS worker is missing",
                            ));
                        }
                        Command::new(worker)
                            .args(["serve", "--instance", instance])
                            .creation_flags(0x08000000)
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .spawn()?;
                        launched = true;
                    }
                }
                Err(e) => return Err(e),
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Teamy TTS worker startup timed out",
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        pipe.enqueue(Frame::empty(HELLO, 0))?;
        let mut ping = Instant::now();
        loop {
            if !tick() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "speech cancelled",
                ));
            }
            for frame in pipe.pump()? {
                match frame.kind {
                    STATUS => {
                        let status: Status = frame.parse()?;
                        if status.state == "ready" {
                            return Ok(Self {
                                pipe,
                                next_id: 1,
                                pid: status.pid,
                            });
                        }
                        if status.state == "failed" {
                            return Err(io::Error::other(status.detail));
                        }
                    }
                    ERROR => {
                        return Err(io::Error::other(
                            String::from_utf8_lossy(&frame.payload).into_owned(),
                        ));
                    }
                    _ => return Err(invalid("unexpected startup response")),
                }
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Teamy TTS model loading timed out",
                ));
            }
            if ping.elapsed() >= Duration::from_millis(100) {
                pipe.enqueue(Frame::empty(PING, 0))?;
                ping = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    pub fn synthesize(
        &mut self,
        request: &Request,
        mut tick: impl FnMut() -> bool,
    ) -> io::Result<Vec<i16>> {
        request.validate()?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| invalid("request id exhausted"))?;
        self.pipe.enqueue(Frame::json(SYNTHESIZE, id, request)?)?;
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut audio = Vec::new();
        loop {
            if !tick() {
                self.pipe.enqueue(Frame::empty(CANCEL, id))?;
                let _ = self.pipe.pump();
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "speech cancelled",
                ));
            }
            for frame in self.pipe.pump()? {
                if frame.id != id {
                    continue;
                }
                match frame.kind {
                    AUDIO => {
                        if frame.payload.len() % 2 != 0
                            || audio.len() * 2 + frame.payload.len() > MAX_AUDIO_BYTES
                        {
                            return Err(invalid("invalid or oversized PCM"));
                        }
                        audio.extend(
                            frame
                                .payload
                                .chunks_exact(2)
                                .map(|s| i16::from_le_bytes([s[0], s[1]])),
                        );
                    }
                    DONE => return Ok(audio),
                    ERROR => {
                        return Err(io::Error::other(
                            String::from_utf8_lossy(&frame.payload).into_owned(),
                        ));
                    }
                    _ => return Err(invalid("unexpected synthesis response")),
                }
            }
            if Instant::now() >= deadline {
                self.pipe.enqueue(Frame::empty(CANCEL, id))?;
                let _ = self.pipe.pump();
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "Teamy TTS synthesis timed out",
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
