//! Real SAPI host operations used by CLI diagnostics and audio capture.
use crate::{
    WAVE_FORMAT_ID,
    engine::pcm_format,
    registration::{self, wide},
};
use std::{
    path::Path,
    ptr,
    time::{Duration, Instant},
};
use windows::{
    Win32::{Foundation::*, Media::Speech::*, System::Com::*},
    core::*,
};

struct Com;
impl Com {
    fn new() -> Result<Self> {
        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
        }
        Ok(Self)
    }
}
impl Drop for Com {
    fn drop(&mut self) {
        unsafe {
            CoUninitialize();
        }
    }
}
fn token_id(token: &ISpObjectToken) -> Result<String> {
    let id = unsafe { token.GetId()? };
    let result = unsafe { id.to_string() };
    unsafe {
        CoTaskMemFree(Some(id.0.cast()));
    }
    Ok(result?)
}

pub struct VoiceStatus {
    pub enumerated: bool,
    pub default_voice: String,
    pub is_default: bool,
}
pub fn status(machine: bool) -> Result<VoiceStatus> {
    let _com = Com::new()?;
    let target = if machine {
        registration::MACHINE_VOICE_ID
    } else {
        registration::VOICE_ID
    };
    let voice: ISpVoice = unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_INPROC_SERVER)? };
    let default_voice = token_id(&unsafe { voice.GetVoice()? })?;
    let category: ISpObjectTokenCategory =
        unsafe { CoCreateInstance(&SpObjectTokenCategory, None, CLSCTX_INPROC_SERVER)? };
    unsafe {
        category.SetId(SPCAT_VOICES, false)?;
    }
    let tokens = unsafe { category.EnumTokens(PCWSTR::null(), PCWSTR::null())? };
    let mut count = 0;
    unsafe {
        tokens.GetCount(&mut count)?;
    }
    let mut enumerated = false;
    for i in 0..count {
        enumerated |= token_id(&unsafe { tokens.Item(i)? })?.eq_ignore_ascii_case(target);
    }
    Ok(VoiceStatus {
        enumerated,
        is_default: default_voice.eq_ignore_ascii_case(target),
        default_voice,
    })
}

/// Explicitly selects Teamy on this voice object, preserving the system default.
pub fn speak(machine: bool, text: &str, output: Option<&Path>) -> Result<Duration> {
    let _com = Com::new()?;
    let voice: ISpVoice = unsafe { CoCreateInstance(&SpVoice, None, CLSCTX_INPROC_SERVER)? };
    let token: ISpObjectToken =
        unsafe { CoCreateInstance(&SpObjectToken, None, CLSCTX_INPROC_SERVER)? };
    let id = wide(if machine {
        registration::MACHINE_VOICE_ID
    } else {
        registration::VOICE_ID
    });
    unsafe {
        token.SetId(PCWSTR::null(), PCWSTR(id.as_ptr()), false)?;
        voice.SetVoice(&token)?;
    }
    let stream = if let Some(output) = output {
        let stream: ISpStream = unsafe { CoCreateInstance(&SpStream, None, CLSCTX_INPROC_SERVER)? };
        let path = wide(
            output
                .to_str()
                .ok_or_else(|| Error::from_hresult(E_INVALIDARG))?,
        );
        unsafe {
            stream.BindToFile(
                PCWSTR(path.as_ptr()),
                SPFM_CREATE_ALWAYS,
                Some(&WAVE_FORMAT_ID),
                Some(&pcm_format()),
                0,
            )?;
            voice.SetOutput(&stream, false)?;
        }
        Some(stream)
    } else {
        None
    };
    let text = wide(text);
    let start = Instant::now();
    unsafe {
        voice.Speak(PCWSTR(text.as_ptr()), 17, None)?;
    }
    let done = unsafe { (voice.vtable().WaitUntilDone)(voice.as_raw(), 90_000) };
    if done != S_OK {
        unsafe {
            let _ = voice.Speak(PCWSTR::null(), 3, None);
        }
        done.ok()?;
        return Err(Error::new(E_FAIL, "SAPI test exceeded 90 seconds"));
    }
    let mut status = SPVOICESTATUS::default();
    unsafe {
        voice.GetStatus(&mut status, ptr::null_mut())?;
    }
    status.hrLastResult.ok()?;
    drop(voice);
    if let Some(stream) = stream {
        unsafe {
            stream.Close()?;
        }
    }
    Ok(start.elapsed())
}
