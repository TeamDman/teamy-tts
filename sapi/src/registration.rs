//! Explicit machine/per-user registration. Never changes the default voice.
use std::path::Path;
use windows::{
    Win32::{Foundation::*, System::Registry::*},
    core::*,
};

pub const CLASS_KEY: &str = r"Software\Classes\CLSID\{3D190E91-1F23-4BF5-A136-36620C2E406E}";
pub const TEST_CLASS_KEY: &str = r"Software\Classes\CLSID\{11B2F305-8AAB-4360-9E2F-CC49D3535B2A}";
pub const VOICE_KEY: &str = r"Software\Microsoft\Speech\Voices\Tokens\TeamyTts";
pub const VOICE_ID: &str = r"HKEY_CURRENT_USER\Software\Microsoft\Speech\Voices\Tokens\TeamyTts";
pub const MACHINE_VOICE_ID: &str =
    r"HKEY_LOCAL_MACHINE\Software\Microsoft\Speech\Voices\Tokens\TeamyTts";
pub const TEST_VOICE_KEY: &str = r"Software\Microsoft\Speech\Voices\Tokens\TeamyTtsTest";
pub const TEST_VOICE_ID: &str =
    r"HKEY_CURRENT_USER\Software\Microsoft\Speech\Voices\Tokens\TeamyTtsTest";

pub fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(Some(0)).collect()
}

struct Key(HKEY);
impl Drop for Key {
    fn drop(&mut self) {
        unsafe {
            let _ = RegCloseKey(self.0);
        }
    }
}

#[cfg(feature = "test-fixture")]
fn set(key: &str, name: &str, value: &str) -> Result<()> {
    set_in(HKEY_CURRENT_USER, key, name, value)
}
fn set_in(hive: HKEY, key: &str, name: &str, value: &str) -> Result<()> {
    let mut handle = HKEY::default();
    let path = wide(key);
    unsafe {
        RegCreateKeyExW(
            hive,
            PCWSTR(path.as_ptr()),
            None,
            PCWSTR::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE | KEY_WOW64_64KEY,
            None,
            &mut handle,
            None,
        )
        .ok()?;
    }
    let key = Key(handle);
    let name = wide(name);
    let data: Vec<u8> = wide(value).iter().flat_map(|x| x.to_le_bytes()).collect();
    unsafe { RegSetValueExW(key.0, PCWSTR(name.as_ptr()), None, REG_SZ, Some(&data)).ok() }
}

pub fn install(dll: &Path, worker: &Path, test: bool) -> Result<()> {
    install_scope(dll, worker, test, false)
}

pub fn install_scope(dll: &Path, worker: &Path, test: bool, machine: bool) -> Result<()> {
    let hive = if machine {
        HKEY_LOCAL_MACHINE
    } else {
        HKEY_CURRENT_USER
    };
    let set = |key: &str, name: &str, value: &str| set_in(hive, key, name, value);
    if !dll.is_absolute()
        || !dll.is_file()
        || (!test && (!worker.is_absolute() || !worker.is_file()))
    {
        return Err(E_INVALIDARG.into());
    }
    let dll = dll
        .to_str()
        .ok_or_else(|| Error::from_hresult(E_INVALIDARG))?;
    let worker = worker
        .to_str()
        .ok_or_else(|| Error::from_hresult(E_INVALIDARG))?;
    let class = if test { TEST_CLASS_KEY } else { CLASS_KEY };
    set(&format!(r"{class}\InprocServer32"), "", dll)?;
    set(
        &format!(r"{class}\InprocServer32"),
        "ThreadingModel",
        "Both",
    )?;
    let key = if test { TEST_VOICE_KEY } else { VOICE_KEY };
    set(
        key,
        "",
        if test {
            "Teamy TTS test fixture"
        } else {
            "Teamy GLaDOS (native)"
        },
    )?;
    set(
        key,
        "CLSID",
        if test {
            "{11B2F305-8AAB-4360-9E2F-CC49D3535B2A}"
        } else {
            "{3D190E91-1F23-4BF5-A136-36620C2E406E}"
        },
    )?;
    set(key, "WorkerPath", worker)?;
    set(key, "TestFixture", if test { "1" } else { "0" })?;
    for (name, value) in [
        ("Language", "409"),
        ("Gender", "Female"),
        ("Age", "Adult"),
        ("Vendor", "Teamy"),
        ("Name", "Teamy GLaDOS"),
    ] {
        set(&format!(r"{key}\Attributes"), name, value)?;
    }
    Ok(())
}

fn delete(hive: HKEY, key: &str) -> Result<()> {
    let path = wide(key);
    let result = unsafe { RegDeleteTreeW(hive, PCWSTR(path.as_ptr())) };
    if result == ERROR_FILE_NOT_FOUND {
        Ok(())
    } else {
        result.ok()
    }
}

pub fn uninstall(test: bool, remove_class: bool) -> Result<()> {
    uninstall_scope(test, remove_class, false)
}
pub fn uninstall_scope(test: bool, remove_class: bool, machine: bool) -> Result<()> {
    let hive = if machine {
        HKEY_LOCAL_MACHINE
    } else {
        HKEY_CURRENT_USER
    };
    delete(hive, if test { TEST_VOICE_KEY } else { VOICE_KEY })?;
    if remove_class {
        delete(hive, if test { TEST_CLASS_KEY } else { CLASS_KEY })?;
    }
    Ok(())
}

pub struct Installed {
    pub dll: String,
    pub worker: String,
}
pub fn installed(machine: bool) -> Result<Option<Installed>> {
    let hive = if machine {
        HKEY_LOCAL_MACHINE
    } else {
        HKEY_CURRENT_USER
    };
    let Some(worker) = read(hive, VOICE_KEY, "WorkerPath")? else {
        return Ok(None);
    };
    let dll = read(hive, &format!(r"{CLASS_KEY}\InprocServer32"), "")?.unwrap_or_default();
    Ok(Some(Installed { dll, worker }))
}

fn read(hive: HKEY, key: &str, name: &str) -> Result<Option<String>> {
    let key = wide(key);
    let name = wide(name);
    let mut bytes = 0;
    let result = unsafe {
        RegGetValueW(
            hive,
            PCWSTR(key.as_ptr()),
            PCWSTR(name.as_ptr()),
            RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
            None,
            None,
            Some(&mut bytes),
        )
    };
    if result == ERROR_FILE_NOT_FOUND {
        return Ok(None);
    }
    result.ok()?;
    if bytes == 0 || bytes > 65536 || bytes % 2 != 0 {
        return Err(E_INVALIDARG.into());
    }
    let mut data = vec![0u16; bytes as usize / 2];
    unsafe {
        RegGetValueW(
            hive,
            PCWSTR(key.as_ptr()),
            PCWSTR(name.as_ptr()),
            RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
            None,
            Some(data.as_mut_ptr().cast()),
            Some(&mut bytes),
        )
        .ok()?;
    }
    let len = data.iter().position(|v| *v == 0).unwrap_or(data.len());
    Ok(Some(
        String::from_utf16(&data[..len]).map_err(|_| Error::from_hresult(E_INVALIDARG))?,
    ))
}

#[cfg(feature = "test-fixture")]
pub fn test_native_worker(instance: &str) -> Result<()> {
    set(TEST_VOICE_KEY, "TestFixture", "0")?;
    set(TEST_VOICE_KEY, "WorkerInstance", instance)
}
