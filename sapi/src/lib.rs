#![cfg(windows)]
#![allow(non_snake_case)]

mod engine;
pub mod host;
pub mod registration;
pub mod text;

use std::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};
use windows::{
    Win32::{Foundation::*, System::Com::*},
    core::*,
};

pub const ENGINE_CLSID: GUID = GUID::from_u128(0x3d190e91_1f23_4bf5_a136_36620c2e406e);
pub const TEST_CLSID: GUID = GUID::from_u128(0x11b2f305_8aab_4360_9e2f_cc49d3535b2a);
pub const WAVE_FORMAT_ID: GUID = GUID::from_u128(0xc31adbae_527f_4ff5_a230_f62bb61ff70c);
static OBJECTS: AtomicUsize = AtomicUsize::new(0);
static LOCKS: AtomicUsize = AtomicUsize::new(0);

struct Lifetime;
impl Lifetime {
    fn new() -> Self {
        OBJECTS.fetch_add(1, Ordering::SeqCst);
        Self
    }
}
impl Drop for Lifetime {
    fn drop(&mut self) {
        OBJECTS.fetch_sub(1, Ordering::SeqCst);
    }
}

fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(Error::from_hresult(E_FAIL)))
}

#[implement(IClassFactory)]
struct Factory {
    _life: Lifetime,
}
impl IClassFactory_Impl for Factory_Impl {
    fn CreateInstance(
        &self,
        outer: Ref<IUnknown>,
        iid: *const GUID,
        out: *mut *mut c_void,
    ) -> Result<()> {
        guarded(|| {
            if out.is_null() || iid.is_null() {
                return Err(E_POINTER.into());
            }
            // COM owns the output reference only after successful QueryInterface.
            unsafe {
                out.write(ptr::null_mut());
            }
            if outer.is_some() {
                return Err(CLASS_E_NOAGGREGATION.into());
            }
            let engine: IUnknown = engine::Engine::new().into();
            unsafe { engine.query(iid, out).ok() }
        })
    }
    fn LockServer(&self, lock: BOOL) -> Result<()> {
        if lock.as_bool() {
            LOCKS.fetch_add(1, Ordering::SeqCst);
        } else {
            let _ = LOCKS.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
        }
        Ok(())
    }
}

/// COM loader entry point. The caller supplies valid GUID/output pointers.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllGetClassObject(
    class: *const GUID,
    iid: *const GUID,
    out: *mut *mut c_void,
) -> HRESULT {
    guarded(|| {
        if class.is_null() || iid.is_null() || out.is_null() {
            return Err(E_POINTER.into());
        }
        unsafe {
            out.write(ptr::null_mut());
        }
        let class = unsafe { *class };
        if class != ENGINE_CLSID && !(cfg!(feature = "test-fixture") && class == TEST_CLSID) {
            return Err(CLASS_E_CLASSNOTAVAILABLE.into());
        }
        let factory: IClassFactory = Factory {
            _life: Lifetime::new(),
        }
        .into();
        unsafe { factory.query(iid, out).ok() }
    })
    .into()
}

#[unsafe(no_mangle)]
pub extern "system" fn DllCanUnloadNow() -> HRESULT {
    if OBJECTS.load(Ordering::SeqCst) == 0 && LOCKS.load(Ordering::SeqCst) == 0 {
        S_OK
    } else {
        S_FALSE
    }
}
