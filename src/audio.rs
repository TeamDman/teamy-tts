//! Local audio playback adapters.

use eyre::bail;
/// Play an in-memory WAV buffer synchronously through the operating system
/// audio device.
///
/// # Errors
///
/// Returns an error when the platform cannot start playback or does not have
/// a built-in playback adapter in this build.
#[cfg(windows)]
pub fn play_wav_bytes(wav: &[u8]) -> eyre::Result<()> {
    use windows::Win32::Media::Audio::PlaySoundW;
    use windows::Win32::Media::Audio::SND_MEMORY;
    use windows::Win32::Media::Audio::SND_SYNC;
    use windows::core::PCWSTR;

    if wav.is_empty() {
        bail!("cannot play an empty WAV buffer");
    }

    // `PlaySoundW` keeps reading this buffer until synchronous playback
    // completes. The slice remains alive for the entire call, and the cast is
    // required because the Windows API reuses its string-pointer parameter as
    // a byte pointer when `SND_MEMORY` is selected.
    let sound = PCWSTR(wav.as_ptr().cast());
    // SAFETY: `sound` points to a valid in-memory RIFF/WAVE buffer whose
    // lifetime covers this synchronous call.
    let played = unsafe { PlaySoundW(sound, None, SND_MEMORY | SND_SYNC) };
    if !played.as_bool() {
        bail!("Windows could not play in-memory WAV audio");
    }
    Ok(())
}

/// Play an in-memory WAV buffer synchronously through the operating system
/// audio device.
///
/// # Errors
///
/// Returns an actionable error on platforms without the current playback
/// adapter.
#[cfg(not(windows))]
pub fn play_wav_bytes(_wav: &[u8]) -> eyre::Result<()> {
    bail!(
        "in-memory WAV playback is currently implemented only on Windows; use `write` to save the audio"
    );
}
