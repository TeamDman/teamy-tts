//! Local audio playback adapters.

use eyre::bail;
use std::path::Path;

/// Play a WAV file synchronously through the operating system audio device.
///
/// # Errors
///
/// Returns an error when the platform cannot start playback or does not have
/// a built-in playback adapter in this build.
#[cfg(windows)]
pub fn play_wav(path: &Path) -> eyre::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::Media::Audio::PlaySoundW;
    use windows::Win32::Media::Audio::SND_FILENAME;
    use windows::Win32::Media::Audio::SND_SYNC;
    use windows::core::PCWSTR;

    let wide_path = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: `wide_path` is NUL-terminated and remains alive for the
    // synchronous call, so the pointer is valid for the Windows API call.
    let played = unsafe { PlaySoundW(PCWSTR(wide_path.as_ptr()), None, SND_FILENAME | SND_SYNC) };
    if !played.as_bool() {
        bail!("Windows could not play WAV output {}", path.display());
    }
    Ok(())
}

/// Play a WAV file synchronously through the operating system audio device.
///
/// # Errors
///
/// Returns an actionable error on platforms without the current playback
/// adapter.
#[cfg(not(windows))]
pub fn play_wav(path: &Path) -> eyre::Result<()> {
    bail!(
        "WAV playback is currently implemented only on Windows; generated output is at {}",
        path.display()
    );
}
