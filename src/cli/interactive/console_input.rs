//! Keep Windows console editing while allowing Ctrl-D to finish a read.

use std::io;
use windows::Win32::System::Console::{
    CONSOLE_READCONSOLE_CONTROL, GetStdHandle, ReadConsoleW, STD_INPUT_HANDLE,
};

pub(super) fn read_line(line: &mut String) -> io::Result<usize> {
    // SAFETY: this retrieves the process's borrowed stdin handle. The caller
    // checked that stdin is a terminal; ownership stays with the process.
    let input = unsafe { GetStdHandle(STD_INPUT_HANDLE) }.map_err(io::Error::other)?;
    let mut control = CONSOLE_READCONSOLE_CONTROL {
        nLength: size_of::<CONSOLE_READCONSOLE_CONTROL>() as u32,
        dwCtrlWakeupMask: 1 << 4,
        ..Default::default()
    };
    let mut text = Vec::new();
    loop {
        let mut buffer = [0u16; 4096];
        let mut count = 0;
        // SAFETY: input remains valid, buffer holds the requested UTF-16 units,
        // and count/control remain live for the duration of this synchronous call.
        unsafe {
            ReadConsoleW(
                input,
                buffer.as_mut_ptr().cast(),
                buffer.len() as u32,
                &mut count,
                Some(&raw mut control),
            )
        }
        .map_err(io::Error::other)?;
        let chunk = &buffer[..count as usize];
        let ctrl_d = chunk.contains(&4);
        text.extend(chunk.iter().copied().filter(|&unit| unit != 4));
        if ctrl_d {
            eprintln!();
        }
        // Ctrl-D submits pending text, or signals EOF on an empty line, just
        // like canonical Unix input. Preserve Windows Ctrl-Z + Enter as well.
        if text.first() == Some(&26) {
            return Ok(0);
        }
        if ctrl_d || count == 0 || chunk.contains(&10) {
            let decoded = String::from_utf16(&text)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let bytes = decoded.len();
            line.push_str(&decoded);
            return Ok(bytes);
        }
    }
}
