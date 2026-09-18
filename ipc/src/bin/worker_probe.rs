//! Exercises real local pipe framing/lifecycle with the native worker.
use std::{
    io,
    os::windows::process::CommandExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};
use teamy_tts_ipc::{
    client::{Client, status, stop},
    pipe::{Endpoint, Pipe},
    protocol::*,
};

struct WorkerCleanup(String);
impl Drop for WorkerCleanup {
    fn drop(&mut self) {
        let _ = stop(&self.0);
    }
}
struct OwnChild(Child);
impl Drop for OwnChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn spawn(worker: &Path, instance: &str, idle: &str, model: Option<&Path>) -> io::Result<OwnChild> {
    let mut command = Command::new(worker);
    command
        .args(["serve", "--instance", instance, "--idle-seconds", idle])
        .creation_flags(0x08000000)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(model) = model {
        command.env("TEAMY_TTS_NATIVE_MODEL_DIR", model);
    }
    Ok(OwnChild(command.spawn()?))
}
fn wait_status(instance: &str, state: &str) -> io::Result<Status> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Ok(s) = status(instance) {
            if s.state == state {
                return Ok(s);
            }
        }
        if Instant::now() > deadline {
            return Err(io::Error::other("worker state deadline"));
        }
        thread::sleep(Duration::from_millis(10));
    }
}
fn request(text: &str) -> Request {
    Request {
        text: text.into(),
        voice: "p2".into(),
        rate: 0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let worker = PathBuf::from(args.next().ok_or("expected worker path")?).canonicalize()?;
    let receipt = PathBuf::from(args.next().ok_or("expected receipt directory")?);
    std::fs::create_dir_all(&receipt)?;
    let instance = format!("probe-{}", std::process::id());
    let _cleanup = WorkerCleanup(instance.clone());
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let worker = worker.clone();
        let instance = instance.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || -> io::Result<_> {
            barrier.wait();
            let start = Instant::now();
            let mut client = Client::start(&worker, &instance, || true)?;
            let loaded_ms = start.elapsed().as_millis();
            let audio = client.synthesize(&request("Hello, friend"), || true)?;
            Ok((client.pid, loaded_ms, audio))
        }));
    }
    let a = handles.remove(0).join().unwrap()?;
    let b = handles.remove(0).join().unwrap()?;
    assert_eq!(a.0, b.0, "racing starters must reach the same worker");
    assert_eq!(a.2, b.2, "concurrent synthesis changed native PCM");
    assert!(a.2.len() > 1000);
    let bytes: Vec<u8> = a.2.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(receipt.join("hello.pcm"), bytes)?;

    let mut client = Client::start(&worker, &instance, || true)?;
    let start = Instant::now();
    let warm = client.synthesize(&request("Hello, friend"), || true)?;
    let warm_ms = start.elapsed().as_millis();
    assert_eq!(warm, a.2);
    let start = Instant::now();
    let long = "The quick brown fox jumps over the lazy dog. ".repeat(6);
    let error = client
        .synthesize(&request(&long), || {
            start.elapsed() < Duration::from_millis(20)
        })
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    let cancel_ms = start.elapsed().as_millis();
    assert!(cancel_ms < 100);
    let start = Instant::now();
    let response = status(&instance)?;
    let control_ms = start.elapsed().as_millis();
    assert_eq!(response.state, "ready");
    assert!(control_ms < 100, "control path blocked on inference");
    let recovered = client.synthesize(&request("Hello, friend"), || true)?;
    assert_eq!(
        recovered, warm,
        "late cancelled audio contaminated next request"
    );

    // One client's cancellation cannot cancel another client's accepted work.
    let other_worker = worker.clone();
    let other_instance = instance.clone();
    let barrier = Arc::new(Barrier::new(2));
    let other_barrier = barrier.clone();
    let other = thread::spawn(move || -> io::Result<_> {
        let mut other = Client::start(&other_worker, &other_instance, || true)?;
        other_barrier.wait();
        other.synthesize(&request("Hello, friend"), || true)
    });
    barrier.wait();
    let start = Instant::now();
    assert_eq!(
        client
            .synthesize(&request(&long), || start.elapsed()
                < Duration::from_millis(20))
            .unwrap_err()
            .kind(),
        io::ErrorKind::Interrupted
    );
    assert_eq!(other.join().unwrap()?, warm);
    let mut fast = request("Hello, friend");
    fast.rate = 10;
    let fast = client.synthesize(&fast, || true)?;
    assert!(
        fast.len() < warm.len() * 3 / 4 && fast.len() > warm.len() / 4,
        "native rate has no effect"
    );

    // A burst cannot create an unbounded per-connection request queue.
    let mut burst = Pipe::connect(&Endpoint::current(&instance)?)?;
    burst.enqueue(Frame::empty(HELLO, 0))?;
    for id in 1..=32 {
        burst.enqueue(Frame::json(SYNTHESIZE, id, &request(&long))?)?;
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut rejected = 0;
    while rejected < 31 {
        for frame in burst.pump()? {
            if frame.kind == ERROR {
                rejected += 1;
            }
        }
        assert!(Instant::now() < deadline, "burst was not bounded");
        thread::sleep(Duration::from_millis(5));
    }
    drop(burst);

    let mut malformed = Pipe::connect(&Endpoint::current(&instance)?)?;
    malformed.enqueue(Frame::empty(HELLO, 0))?;
    malformed.enqueue(Frame {
        kind: SYNTHESIZE,
        id: 1,
        payload: br#"{"text":"bad","voice":"invalid","rate":0}"#.to_vec(),
    })?;
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if malformed.pump().is_err() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "malformed client not disconnected"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        client.synthesize(&request("Hello, friend"), || true)?,
        warm,
        "one bad client affected another"
    );
    drop(client);
    stop(&instance)?;
    let crash_instance = format!("{instance}-crash");
    let _crash_cleanup = WorkerCleanup(crash_instance.clone());
    let mut child = spawn(&worker, &crash_instance, "120", None)?;
    assert_eq!(wait_status(&crash_instance, "ready")?.pid, child.0.id());
    child.0.kill()?;
    child.0.wait()?;
    let mut restarted = Client::start(&worker, &crash_instance, || true)?;
    assert_ne!(restarted.pid, child.0.id());
    assert_eq!(
        restarted.synthesize(&request("Hello, friend"), || true)?,
        warm
    );
    drop(restarted);
    stop(&crash_instance)?;

    let idle_instance = format!("{instance}-idle");
    let mut idle = spawn(&worker, &idle_instance, "1", None)?;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(exit) = idle.0.try_wait()? {
            assert!(exit.success());
            break;
        }
        assert!(Instant::now() < deadline, "idle worker did not exit");
        thread::sleep(Duration::from_millis(20));
    }
    let failed_instance = format!("{instance}-failed");
    let _failed = spawn(
        &worker,
        &failed_instance,
        "120",
        Some(&receipt.join("missing-model")),
    )?;
    let failed = wait_status(&failed_instance, "failed")?;
    assert!(!failed.detail.is_empty());
    assert!(Client::start(&worker, &failed_instance, || true).is_err());
    stop(&failed_instance)?;
    let receipt_text = format!(
        "{{\"singleton_pid\":{},\"cold_ready_ms\":[{},{}],\"warm_ms\":{warm_ms},\"cancel_ms\":{cancel_ms},\"control_ms\":{control_ms},\"samples\":{},\"same_pcm\":true,\"malformed_isolated\":true,\"cancellation_isolated\":true,\"native_rate\":true,\"burst_rejected\":{rejected},\"restart\":true,\"idle_exit\":true,\"missing_model_error\":true,\"stopped\":true}}\n",
        a.0,
        a.1,
        b.1,
        warm.len()
    );
    std::fs::write(receipt.join("worker.json"), &receipt_text)?;
    print!("{receipt_text}");
    Ok(())
}
