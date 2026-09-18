use crate::{
    Lifetime, WAVE_FORMAT_ID, guarded,
    registration::wide,
    text::{self, Sentence},
};
use std::{ptr, sync::Mutex};
use windows::{
    Win32::{
        Foundation::*,
        Media::{Audio::WAVEFORMATEX, Speech::*},
        System::Com::*,
    },
    core::*,
};

#[derive(Clone)]
struct Token {
    id: String,
    fixture: bool,
    worker: String,
    instance: String,
}

#[implement(ISpTTSEngine, ISpObjectWithToken)]
pub struct Engine {
    token: Mutex<Option<Token>>,
    _life: Lifetime,
}
impl Engine {
    pub fn new() -> Self {
        Self {
            token: Mutex::new(None),
            _life: Lifetime::new(),
        }
    }
}

pub fn pcm_format() -> WAVEFORMATEX {
    WAVEFORMATEX {
        wFormatTag: 1,
        nChannels: 1,
        nSamplesPerSec: 22050,
        nAvgBytesPerSec: 44100,
        nBlockAlign: 2,
        wBitsPerSample: 16,
        cbSize: 0,
    }
}

unsafe fn com_string(value: PWSTR) -> Result<String> {
    let text = unsafe { value.to_string() };
    unsafe {
        CoTaskMemFree(Some(value.0.cast()));
    }
    text.map_err(Into::into)
}

impl ISpObjectWithToken_Impl for Engine_Impl {
    fn SetObjectToken(&self, token: Ref<ISpObjectToken>) -> Result<()> {
        guarded(|| {
            let token = token.ok()?;
            let id = unsafe { com_string(token.GetId()?)? };
            let fixture = unsafe {
                token
                    .GetStringValue(w!("TestFixture"))
                    .and_then(|s| com_string(s))
                    .unwrap_or_default()
                    == "1"
            };
            let worker = unsafe {
                token
                    .GetStringValue(w!("WorkerPath"))
                    .and_then(|s| com_string(s))?
            };
            let instance = unsafe {
                token
                    .GetStringValue(w!("WorkerInstance"))
                    .and_then(|s| com_string(s))
                    .unwrap_or_else(|_| "main".into())
            };
            *self.token.lock().map_err(|_| Error::from_hresult(E_FAIL))? = Some(Token {
                id,
                fixture,
                worker,
                instance,
            });
            Ok(())
        })
    }
    fn GetObjectToken(&self) -> Result<ISpObjectToken> {
        guarded(|| {
            let token = self
                .token
                .lock()
                .map_err(|_| Error::from_hresult(E_FAIL))?
                .clone()
                .ok_or_else(|| Error::from_hresult(E_UNEXPECTED))?;
            let result: ISpObjectToken =
                unsafe { CoCreateInstance(&SpObjectToken, None, CLSCTX_INPROC_SERVER)? };
            let id = wide(&token.id);
            unsafe {
                result.SetId(PCWSTR::null(), PCWSTR(id.as_ptr()), false)?;
            }
            Ok(result)
        })
    }
}

impl ISpTTSEngine_Impl for Engine_Impl {
    fn GetOutputFormat(
        &self,
        _: *const GUID,
        _: *const WAVEFORMATEX,
        id: *mut GUID,
        out: *mut *mut WAVEFORMATEX,
    ) -> Result<()> {
        guarded(|| {
            if id.is_null() || out.is_null() {
                return Err(E_POINTER.into());
            }
            unsafe {
                out.write(ptr::null_mut());
            }
            let format =
                unsafe { CoTaskMemAlloc(size_of::<WAVEFORMATEX>()).cast::<WAVEFORMATEX>() };
            if format.is_null() {
                return Err(E_OUTOFMEMORY.into());
            }
            unsafe {
                format.write(pcm_format());
                id.write(WAVE_FORMAT_ID);
                out.write(format);
            }
            Ok(())
        })
    }
    fn Speak(
        &self,
        _: u32,
        id: *const GUID,
        format: *const WAVEFORMATEX,
        fragments: *const SPVTEXTFRAG,
        site: Ref<ISpTTSEngineSite>,
    ) -> Result<()> {
        guarded(|| {
            if id.is_null() || format.is_null() {
                return Err(E_POINTER.into());
            }
            let f = unsafe { *format };
            if unsafe { *id } != WAVE_FORMAT_ID
                || f.wFormatTag != 1
                || f.nChannels != 1
                || f.nSamplesPerSec != 22050
                || f.wBitsPerSample != 16
                || f.nBlockAlign != 2
            {
                return Err(E_INVALIDARG.into());
            }
            let token = self
                .token
                .lock()
                .map_err(|_| Error::from_hresult(E_FAIL))?
                .clone()
                .ok_or_else(|| Error::from_hresult(E_UNEXPECTED))?;
            let units = unsafe { copy_fragments(fragments)? };
            let site = site.ok()?;
            speak(&token, site, &units)
        })
    }
}

struct Unit {
    sentence: Sentence,
    silence_ms: u32,
}
unsafe fn copy_fragments(mut fragment: *const SPVTEXTFRAG) -> Result<Vec<Unit>> {
    let mut units = Vec::new();
    let mut total = 0usize;
    let mut count = 0;
    while !fragment.is_null() {
        count += 1;
        if count > 1024 {
            return Err(E_INVALIDARG.into());
        }
        let f = unsafe { &*fragment };
        total = total
            .checked_add(f.ulTextLen as usize)
            .ok_or_else(|| Error::from_hresult(E_INVALIDARG))?;
        if total > text::MAX_TEXT_UTF16 {
            return Err(E_INVALIDARG.into());
        }
        if f.State.eAction == SPVA_Silence {
            units.push(Unit {
                sentence: Sentence {
                    text: String::new(),
                    offset: f.ulTextSrcOffset,
                    len: 0,
                },
                silence_ms: f.State.SilenceMSecs.min(10_000),
            });
        } else if f.ulTextLen != 0 {
            if f.pTextStart.is_null() {
                return Err(E_POINTER.into());
            }
            let raw = unsafe { std::slice::from_raw_parts(f.pTextStart.0, f.ulTextLen as usize) };
            let value = String::from_utf16(raw).map_err(|_| Error::from_hresult(E_INVALIDARG))?;
            units.extend(
                text::sentences(&value, f.ulTextSrcOffset)
                    .into_iter()
                    .map(|sentence| Unit {
                        sentence,
                        silence_ms: 0,
                    }),
            );
        }
        fragment = f.pNext;
    }
    Ok(units)
}

enum Action {
    Continue,
    Jump(usize),
    Stop,
}
fn actions(site: &ISpTTSEngineSite, index: usize, count: usize) -> Result<Action> {
    let flags = unsafe { site.GetActions() };
    if flags & SPVES_ABORT.0 as u32 != 0 {
        return Ok(Action::Stop);
    }
    if flags & SPVES_SKIP.0 as u32 != 0 {
        let mut kind = SPVSKIPTYPE::default();
        let mut requested = 0;
        unsafe {
            site.GetSkipInfo(&mut kind, &mut requested)?;
        }
        if kind != SPVST_SENTENCE {
            unsafe {
                site.CompleteSkip(0)?;
            }
            return Ok(Action::Stop);
        }
        let target = (index as i64 + i64::from(requested)).clamp(0, count as i64) as usize;
        let skipped = target as i32 - index as i32;
        unsafe {
            site.CompleteSkip(skipped)?;
        }
        if skipped != requested || target == count {
            return Ok(Action::Stop);
        }
        return Ok(Action::Jump(target));
    }
    Ok(Action::Continue)
}

fn speak(token: &Token, site: &ISpTTSEngineSite, units: &[Unit]) -> Result<()> {
    use teamy_tts_ipc::{client::Client, protocol::Request};
    let mut index = 0;
    let mut offset = 0u64;
    let mut client: Option<Client> = None;
    'sentences: while index < units.len() {
        match actions(site, index, units.len())? {
            Action::Stop => return Ok(()),
            Action::Jump(i) => {
                index = i;
                continue;
            }
            Action::Continue => {}
        }
        let unit = &units[index];
        // The event position uses original UTF-16 source offsets, not UTF-8 bytes.
        let event = SPEVENT {
            _bitfield: SPEI_SENTENCE_BOUNDARY.0,
            ulStreamNum: 0,
            ullAudioStreamOffset: offset,
            wParam: WPARAM(unit.sentence.len as usize),
            lParam: LPARAM(unit.sentence.offset as isize),
        };
        unsafe {
            site.AddEvents(&event, 1)?;
        }
        let chunks = if unit.silence_ms > 0 {
            vec![""]
        } else {
            text::chunks(&unit.sentence.text)
        };
        for text in chunks {
            let rate = unsafe { site.GetRate()? };
            let mut pending = None;
            let mut poll = || match actions(site, index, units.len()) {
                Ok(Action::Continue) => true,
                value => {
                    pending = Some(value);
                    false
                }
            };
            let output = if unit.silence_ms > 0 {
                Ok(vec![0i16; unit.silence_ms as usize * 22050 / 1000])
            } else if cfg!(feature = "test-fixture") && token.fixture {
                fixture(text, rate)
            } else {
                (|| {
                    if client.is_none() {
                        client = Some(Client::start(
                            std::path::Path::new(&token.worker),
                            &token.instance,
                            &mut poll,
                        )?);
                    }
                    client.as_mut().unwrap().synthesize(
                        &Request {
                            text: text.into(),
                            voice: "p2".into(),
                            rate,
                        },
                        &mut poll,
                    )
                })()
            };
            if let Some(action) = pending {
                match action? {
                    Action::Stop => return Ok(()),
                    Action::Jump(i) => {
                        index = i;
                        continue 'sentences;
                    }
                    Action::Continue => {}
                }
            }
            let samples = output.map_err(|error| Error::new(E_FAIL, error.to_string()))?;
            for chunk in samples.chunks(220) {
                match actions(site, index, units.len())? {
                    Action::Stop => return Ok(()),
                    Action::Jump(i) => {
                        index = i;
                        continue 'sentences;
                    }
                    Action::Continue => {}
                }
                let volume = unsafe { site.GetVolume()? }.min(100) as i32;
                let bytes: Vec<u8> = chunk
                    .iter()
                    .flat_map(|s| ((*s as i32 * volume / 100) as i16).to_le_bytes())
                    .collect();
                let mut written = 0;
                while written < bytes.len() {
                    let mut count = 0;
                    let hr = unsafe {
                        (site.vtable().Write)(
                            site.as_raw(),
                            bytes[written..].as_ptr().cast(),
                            (bytes.len() - written) as u32,
                            &mut count,
                        )
                    };
                    hr.ok()?;
                    // SP_AUDIO_STOPPED is a success code: preserve it instead of losing it in Result<()>.
                    if hr != S_OK {
                        return Ok(());
                    }
                    if count == 0 || count as usize > bytes.len() - written || count % 2 != 0 {
                        return Err(E_FAIL.into());
                    }
                    written += count as usize;
                    offset += u64::from(count);
                }
                #[cfg(feature = "test-fixture")]
                if token.fixture {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
        index += 1;
    }
    Ok(())
}

fn fixture(text: &str, rate: i32) -> std::io::Result<Vec<i16>> {
    #[cfg(feature = "test-fixture")]
    {
        let marker = text.as_bytes().first().copied().unwrap_or(b'X') as i16 * 100;
        return Ok(vec![
            marker;
            (22050.0 / 2f64.powf(rate as f64 / 10.0)) as usize
        ]);
    }
    #[cfg(not(feature = "test-fixture"))]
    {
        let _ = (text, rate);
        Err(std::io::Error::other(
            "fixture is unavailable in a production build",
        ))
    }
}
