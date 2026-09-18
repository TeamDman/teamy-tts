//! Responsive pipe control plus a single thread that owns all native CUDA state.
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use teamy_cancellation::CancellationToken;
use teamy_tts_ipc::{
    pipe::{Endpoint, Pipe},
    protocol::*,
};

struct Job {
    client: u64,
    id: u64,
    request: Request,
    cancelled: Arc<AtomicBool>,
}
enum Event {
    Ready(Result<(), String>),
    Audio {
        client: u64,
        id: u64,
        result: Result<Vec<u8>, String>,
    },
}
struct Connection {
    pipe: Pipe,
    hello: bool,
    last_id: u64,
    active: Option<(u64, Arc<AtomicBool>)>,
    touched: Instant,
}
impl Connection {
    fn cancel(&mut self) {
        if let Some((_, cancelled)) = self.active.take() {
            cancelled.store(true, Ordering::Relaxed);
        }
    }
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.cancel();
    }
}

pub fn serve(
    instance: &str,
    idle_seconds: u32,
    cancellation: CancellationToken,
) -> eyre::Result<()> {
    let endpoint = Endpoint::current(instance)?;
    // Claim the singleton before allocating GPU resources. A competing starter
    // exits here; clients continue connecting to the process that won the claim.
    let mut listener = Pipe::listen(&endpoint, true)
        .map_err(|e| eyre::eyre!("cannot claim worker instance {instance}: {e}"))?;
    let model_dir = crate::runtime::configured_model_dir();
    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(8);
    let (events_tx, events_rx) = mpsc::sync_channel(16);
    let inference = thread::Builder::new()
        .name("teamy-tts-inference".into())
        .spawn(move || {
            let runtime =
                model_dir.and_then(|dir| crate::runtime::GladosRuntime::from_native(&dir));
            let runtime = match runtime {
                Ok(runtime) => {
                    if events_tx.send(Event::Ready(Ok(()))).is_err() {
                        return;
                    }
                    runtime
                }
                Err(error) => {
                    let _ = events_tx.send(Event::Ready(Err(format!("{error:#}"))));
                    return;
                }
            };
            while let Ok(job) = jobs_rx.recv() {
                if job.cancelled.load(Ordering::Relaxed) {
                    continue;
                }
                // The native alpha parameter increases speaking speed as it rises.
                let alpha = 2f32.powf(job.request.rate as f32 / 10.0);
                let result = runtime
                    .synthesize(&job.request.text, &job.request.voice, alpha)
                    .and_then(|audio| {
                        eyre::ensure!(
                            audio.len() * 2 <= MAX_AUDIO_BYTES,
                            "native output exceeds the 60-second unit limit"
                        );
                        eyre::ensure!(
                            audio.iter().all(|s| s.is_finite()),
                            "native output contains non-finite samples"
                        );
                        Ok(audio
                            .iter()
                            .flat_map(|s| {
                                ((s.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes()
                            })
                            .collect())
                    })
                    .map_err(|e| format!("{e:#}"));
                if job.cancelled.load(Ordering::Relaxed) {
                    continue;
                }
                if events_tx
                    .send(Event::Audio {
                        client: job.client,
                        id: job.id,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
            }
        })?;
    let result = control(
        &endpoint,
        &mut listener,
        &jobs_tx,
        &events_rx,
        idle_seconds,
        &cancellation,
    );
    // Releasing queued work and every client cancels only this worker's jobs.
    drop(listener);
    drop(jobs_tx);
    drop(events_rx);
    inference
        .join()
        .map_err(|_| eyre::eyre!("native inference thread panicked"))?;
    result
}

fn control(
    endpoint: &Endpoint,
    listener: &mut Pipe,
    jobs: &mpsc::SyncSender<Job>,
    events: &mpsc::Receiver<Event>,
    idle_seconds: u32,
    cancellation: &CancellationToken,
) -> eyre::Result<()> {
    let mut clients = BTreeMap::<u64, Connection>::new();
    let mut next_client = 1;
    let mut status = Status {
        state: "loading".into(),
        pid: std::process::id(),
        version: env!("GIT_REVISION").into(),
        detail: String::new(),
    };
    let mut idle_since = Instant::now();
    let mut failed_since = None;
    let mut stop = false;
    tracing::info!(instance = %endpoint.name, "native worker listening");
    while !stop && !cancellation.is_cancelled() {
        if clients.len() < 16 && listener.accept()? {
            let replacement = Pipe::listen(endpoint, false)?;
            let pipe = std::mem::replace(listener, replacement);
            clients.insert(
                next_client,
                Connection {
                    pipe,
                    hello: false,
                    last_id: 0,
                    active: None,
                    touched: Instant::now(),
                },
            );
            next_client += 1;
            idle_since = Instant::now();
        }
        loop {
            let event = match events.try_recv() {
                Ok(event) => event,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if status.state != "failed" {
                        status.state = "failed".into();
                        status.detail = "native inference thread stopped unexpectedly".into();
                        failed_since = Some(Instant::now());
                        for connection in clients.values_mut() {
                            if let Some((id, _)) = connection.active.as_ref() {
                                let _ = connection.pipe.enqueue(Frame::error(*id, &status.detail));
                            }
                            connection.cancel();
                        }
                    }
                    break;
                }
            };
            match event {
                Event::Ready(result) => match result {
                    Ok(()) => {
                        status.state = "ready".into();
                        tracing::info!("native worker ready");
                    }
                    Err(error) => {
                        status.state = "failed".into();
                        status.detail = error;
                        failed_since = Some(Instant::now());
                    }
                },
                Event::Audio { client, id, result } => {
                    let Some(connection) = clients.get_mut(&client) else {
                        continue;
                    };
                    if !connection
                        .active
                        .as_ref()
                        .is_some_and(|(active, cancelled)| {
                            *active == id && !cancelled.load(Ordering::Relaxed)
                        })
                    {
                        continue;
                    }
                    connection.active = None;
                    let sent: std::io::Result<()> = (|| {
                        match result {
                            Ok(pcm) => {
                                for chunk in pcm.chunks(MAX_PAYLOAD) {
                                    connection.pipe.enqueue(Frame {
                                        kind: AUDIO,
                                        id,
                                        payload: chunk.to_vec(),
                                    })?;
                                }
                                connection.pipe.enqueue(Frame::empty(DONE, id))?;
                            }
                            Err(error) => connection.pipe.enqueue(Frame::error(id, &error))?,
                        }
                        Ok(())
                    })();
                    if sent.is_err() {
                        clients.remove(&client);
                    }
                }
            }
        }
        let mut remove = Vec::new();
        for (&client_id, connection) in &mut clients {
            let frames = match connection.pipe.pump() {
                Ok(frames) => frames,
                Err(_) => {
                    remove.push(client_id);
                    continue;
                }
            };
            for frame in frames {
                connection.touched = Instant::now();
                let result: std::io::Result<()> = (|| {
                    if !connection.hello && frame.kind != HELLO {
                        return Err(invalid("HELLO required"));
                    }
                    match frame.kind {
                        HELLO | PING => {
                            if !frame.payload.is_empty() {
                                return Err(invalid("control payload must be empty"));
                            }
                            connection.hello = true;
                            connection
                                .pipe
                                .enqueue(Frame::json(STATUS, frame.id, &status)?)?;
                        }
                        SYNTHESIZE => {
                            if frame.id == 0 || frame.id <= connection.last_id {
                                return Err(invalid("request ids must increase"));
                            }
                            connection.last_id = frame.id;
                            let request: Request = frame.parse()?;
                            request.validate()?;
                            if status.state != "ready" {
                                connection.pipe.enqueue(Frame::error(
                                    frame.id,
                                    &format!("worker {}: {}", status.state, status.detail),
                                ))?;
                            } else if connection.active.is_some() {
                                connection.pipe.enqueue(Frame::error(
                                    frame.id,
                                    "one synthesis may be outstanding per connection",
                                ))?;
                            } else {
                                let cancelled = Arc::new(AtomicBool::new(false));
                                let job = Job {
                                    client: client_id,
                                    id: frame.id,
                                    request,
                                    cancelled: Arc::clone(&cancelled),
                                };
                                match jobs.try_send(job) {
                                    Ok(()) => connection.active = Some((frame.id, cancelled)),
                                    Err(_) => connection.pipe.enqueue(Frame::error(
                                        frame.id,
                                        "worker synthesis queue is full or unavailable",
                                    ))?,
                                }
                            }
                        }
                        CANCEL => {
                            if connection
                                .active
                                .as_ref()
                                .is_some_and(|(id, _)| *id == frame.id)
                            {
                                connection.cancel();
                            }
                            if connection.last_id == frame.id {
                                connection.pipe.discard_output();
                            }
                        }
                        STOP => {
                            stop = true;
                        }
                        _ => return Err(invalid("unexpected client message")),
                    }
                    Ok(())
                })();
                if result.is_err() {
                    remove.push(client_id);
                    break;
                }
            }
            if connection.touched.elapsed() > Duration::from_secs(70) {
                remove.push(client_id);
            }
        }
        for id in remove {
            clients.remove(&id);
        }
        if !clients.is_empty() {
            idle_since = Instant::now();
        }
        if idle_since.elapsed() >= Duration::from_secs(u64::from(idle_seconds))
            || failed_since.is_some_and(|t: Instant| t.elapsed() > Duration::from_secs(15))
        {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    tracing::info!("native worker shutting down");
    Ok(())
}
