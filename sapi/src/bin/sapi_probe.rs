//! Real SAPI integration probe. Captures audio to files; never plays a test tone.
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use teamy_tts_sapi::{
    WAVE_FORMAT_ID,
    registration::{self, wide},
};
use windows::{
    Win32::{
        Media::{Audio::WAVEFORMATEX, Speech::*},
        System::Com::*,
    },
    core::*,
};

struct Com;
impl Drop for Com {
    fn drop(&mut self) {
        unsafe {
            CoUninitialize();
        }
    }
}
struct Registration;
impl Drop for Registration {
    fn drop(&mut self) {
        let _ = registration::uninstall(true, true);
    }
}

fn token_id(token: &ISpObjectToken) -> Result<String> {
    let id = unsafe { token.GetId()? };
    let text = unsafe { id.to_string() };
    unsafe {
        CoTaskMemFree(Some(id.0.cast()));
    }
    Ok(text?)
}
fn default_voice() -> Result<String> {
    let voice: ISpVoice = unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_INPROC_SERVER)? };
    token_id(&unsafe { voice.GetVoice()? })
}

fn voice(path: &Path) -> Result<(ISpVoice, ISpStream)> {
    let voice: ISpVoice = unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_INPROC_SERVER)? };
    let token: ISpObjectToken =
        unsafe { CoCreateInstance(&SpObjectToken, None, CLSCTX_INPROC_SERVER)? };
    let id = wide(registration::TEST_VOICE_ID);
    unsafe {
        token.SetId(PCWSTR::null(), PCWSTR(id.as_ptr()), false)?;
        voice.SetVoice(&token)?;
    }
    let stream: ISpStream = unsafe { CoCreateInstance(&SpStream, None, CLSCTX_INPROC_SERVER)? };
    let format = WAVEFORMATEX {
        wFormatTag: 1,
        nChannels: 1,
        nSamplesPerSec: 22050,
        nAvgBytesPerSec: 44100,
        nBlockAlign: 2,
        wBitsPerSample: 16,
        cbSize: 0,
    };
    let name = wide(path.to_str().unwrap());
    unsafe {
        stream.BindToFile(
            PCWSTR(name.as_ptr()),
            SPFM_CREATE_ALWAYS,
            Some(&WAVE_FORMAT_ID),
            Some(&format),
            0,
        )?;
        voice.SetOutput(&stream, false)?;
    }
    Ok((voice, stream))
}

fn speak(voice: &ISpVoice, text: &str, flags: u32) -> Result<()> {
    let text = wide(text);
    unsafe { voice.Speak(PCWSTR(text.as_ptr()), flags, None) }
}

fn done(voice: ISpVoice, stream: ISpStream) -> Result<()> {
    // The generated wrapper treats S_FALSE (timeout) as success. Check the HRESULT.
    let hr = unsafe { (voice.vtable().WaitUntilDone)(voice.as_raw(), 10_000) };
    assert_eq!(hr, HRESULT(0), "SAPI speech timed out: {hr:?}");
    let mut status = SPVOICESTATUS::default();
    unsafe {
        voice.GetStatus(&mut status, std::ptr::null_mut())?;
    }
    status.hrLastResult.ok()?;
    drop(voice);
    unsafe {
        stream.Close()?;
    }
    Ok(())
}

fn pcm(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        if &bytes[at..at + 4] == b"data" {
            return bytes[at + 8..at + 8 + size]
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]]))
                .collect();
        }
        at += 8 + size + size % 2;
    }
    panic!("WAV has no data chunk");
}

fn markers(samples: &[i16]) -> Vec<i16> {
    let mut result = Vec::new();
    for &sample in samples {
        if sample != 0 && result.last() != Some(&sample) {
            result.push(sample);
        }
    }
    result
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let dll = PathBuf::from(args.next().ok_or("expected DLL path")?).canonicalize()?;
    let dir = PathBuf::from(args.next().ok_or("expected receipt directory")?);
    let worker = args.next().map(PathBuf::from);
    std::fs::create_dir_all(&dir)?;
    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
    }
    let _com = Com;
    let before = default_voice()?;
    // A test run must never overwrite a production COM registration.
    if std::env::var_os("TEAMY_SAPI_ALLOW_TEST_REGISTRATION").as_deref()
        != Some(std::ffi::OsStr::new("1"))
    {
        return Err(
            "set TEAMY_SAPI_ALLOW_TEST_REGISTRATION=1 in an isolated development run".into(),
        );
    }
    registration::install(
        &dll,
        worker.as_deref().unwrap_or(Path::new("fixture")),
        true,
    )?;
    let _registration = Registration;
    let category: ISpObjectTokenCategory =
        unsafe { CoCreateInstance(&SpObjectTokenCategory, None, CLSCTX_INPROC_SERVER)? };
    unsafe {
        category.SetId(SPCAT_VOICES, false)?;
    }
    let tokens = unsafe { category.EnumTokens(PCWSTR::null(), PCWSTR::null())? };
    let mut found = false;
    let mut token_count = 0;
    unsafe {
        tokens.GetCount(&mut token_count)?;
    }
    for i in 0..token_count {
        let id = token_id(&unsafe { tokens.Item(i)? })?;
        eprintln!("Enumerated voice: {id}");
        if id.eq_ignore_ascii_case(registration::TEST_VOICE_ID) {
            found = true;
        }
    }
    eprintln!("Per-user voice enumerated: {found}");

    if let Some(worker) = worker {
        return native_probe(&worker, &dir, &before);
    }

    let path = dir.join("fifo.wav");
    let (v, s) = voice(&path)?;
    speak(&v, "Alpha.", 17)?;
    speak(&v, "Bravo.", 17)?;
    done(v, s)?;
    assert_eq!(markers(&pcm(&path)), vec![6500, 6600]);

    let path = dir.join("purge.wav");
    let (v, s) = voice(&path)?;
    speak(&v, "Alpha. Bravo.", 17)?;
    std::thread::sleep(Duration::from_millis(200));
    let start = Instant::now();
    speak(&v, "Charlie.", 19)?;
    let purge_ms = start.elapsed().as_millis();
    done(v, s)?;
    let m = markers(&pcm(&path));
    assert!(
        m == vec![6500, 6700] || m == vec![6700],
        "purge markers: {m:?}"
    );
    assert!(purge_ms < 500, "purge blocked for {purge_ms} ms");

    let path = dir.join("skip.wav");
    let (v, s) = voice(&path)?;
    speak(&v, "Alpha. Bravo. Charlie.", 17)?;
    std::thread::sleep(Duration::from_millis(200));
    let start = Instant::now();
    let mut skipped = 0;
    unsafe {
        v.Skip(w!("Sentence"), i32::MAX, &mut skipped)?;
    }
    let skip_ms = start.elapsed().as_millis();
    done(v, s)?;
    let p = pcm(&path);
    assert!(
        markers(&p).iter().all(|s| *s == 6500),
        "stale sentence after Skip"
    );
    assert!(p.len() < 11025, "Skip did not stop current sentence");
    assert!(skip_ms < 500, "Skip blocked for {skip_ms} ms");
    assert_eq!(skipped, 3);

    let path = dir.join("rate-volume.wav");
    let (v, s) = voice(&path)?;
    unsafe {
        v.SetRate(10)?;
        v.SetVolume(50)?;
    }
    speak(&v, "Alpha.", 17)?;
    done(v, s)?;
    let p = pcm(&path);
    assert_eq!(markers(&p), vec![3250]);
    assert_eq!(p.len(), 11025);
    assert_eq!(default_voice()?, before, "default voice changed");
    let receipt = format!(
        "{{\"fixture\":true,\"enumerated\":{found},\"fifo\":true,\"purge_ms\":{purge_ms},\"skip_ms\":{skip_ms},\"skipped\":{skipped},\"rate_volume\":true,\"default_unchanged\":true}}\n"
    );
    std::fs::write(dir.join("fixture.json"), &receipt)?;
    print!("{receipt}");
    Ok(())
}

fn native_probe(
    worker: &Path,
    dir: &Path,
    before: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use teamy_tts_ipc::{client::Client, protocol::*};
    let instance = format!("sapi-probe-{}", std::process::id());
    struct Cleanup(String);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = teamy_tts_ipc::client::stop(&self.0);
        }
    }
    let _cleanup = Cleanup(instance.clone());
    registration::test_native_worker(&instance)?;
    // Purge while the worker/model is still starting: only the replacement may complete.
    let path = dir.join("cold-purge.wav");
    let (v, s) = voice(&path)?;
    let start = Instant::now();
    speak(&v, "An obsolete menu entry.", 17)?;
    std::thread::sleep(Duration::from_millis(20));
    let purge_start = Instant::now();
    speak(&v, "Hello, friend", 19)?;
    let cold_purge_ms = purge_start.elapsed().as_millis();
    done(v, s)?;
    let cold_audio_ms = start.elapsed().as_millis();
    let actual = pcm(&path);
    let mut client = Client::start(worker, &instance, || true)?;
    let expected = client.synthesize(
        &Request {
            text: "Hello, friend".into(),
            voice: "p2".into(),
            rate: 0,
        },
        || true,
    )?;
    assert_eq!(
        actual, expected,
        "cold purge leaked old audio or changed PCM"
    );

    let path = dir.join("native-fifo.wav");
    let (v, s) = voice(&path)?;
    let start = Instant::now();
    speak(&v, "Hello, friend", 17)?;
    speak(&v, "Hello, friend", 17)?;
    done(v, s)?;
    let pair_ms = start.elapsed().as_millis();
    assert_eq!(
        pcm(&path),
        [expected.as_slice(), expected.as_slice()].concat()
    );

    let path = dir.join("native-skip.wav");
    let (v, s) = voice(&path)?;
    let long = "The quick brown fox jumps over the lazy dog. ".repeat(200);
    speak(&v, &long, 17)?;
    std::thread::sleep(Duration::from_millis(10));
    let mut skipped = 0;
    let start = Instant::now();
    unsafe {
        v.Skip(w!("Sentence"), i32::MAX, &mut skipped)?;
    }
    let skip_ms = start.elapsed().as_millis();
    done(v, s)?;
    assert!(
        skip_ms < 150,
        "native synthesis blocked SAPI Skip for {skip_ms} ms"
    );
    assert!(
        pcm(&path).len() < expected.len() * 10,
        "Skip rendered the long utterance"
    );

    let path = dir.join("native-volume.wav");
    let (v, s) = voice(&path)?;
    unsafe {
        v.SetVolume(50)?;
    }
    speak(&v, "Hello, friend", 17)?;
    done(v, s)?;
    let scaled: Vec<i16> = expected
        .iter()
        .map(|s| (*s as i32 * 50 / 100) as i16)
        .collect();
    assert_eq!(pcm(&path), scaled);
    let path = dir.join("native-rate.wav");
    let (v, s) = voice(&path)?;
    unsafe {
        v.SetRate(10)?;
    }
    speak(&v, "Hello, friend", 17)?;
    done(v, s)?;
    let fast = client.synthesize(
        &Request {
            text: "Hello, friend".into(),
            voice: "p2".into(),
            rate: 10,
        },
        || true,
    )?;
    assert_eq!(pcm(&path), fast);
    assert!(fast.len() < expected.len() * 3 / 4);
    assert_eq!(default_voice()?, before);
    let output = format!(
        "{{\"native\":true,\"enumerated\":false,\"pcm_matches_worker\":true,\"cold_audio_ms\":{cold_audio_ms},\"cold_purge_ms\":{cold_purge_ms},\"fifo_pair_ms\":{pair_ms},\"skip_ms\":{skip_ms},\"skipped\":{skipped},\"volume\":true,\"default_unchanged\":true}}\n"
    );
    std::fs::write(dir.join("native.json"), &output)?;
    print!("{output}");
    Ok(())
}
