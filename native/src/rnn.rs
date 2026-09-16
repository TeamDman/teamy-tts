//! Direct cuDNN recurrent inference. Model topology and weights stay in Rust.
//! Signatures follow NVIDIA's cudnn_adv v8 RNN API (supported by cuDNN 9).
use crate::{
    cuda::{Buffer, Device},
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
    rc::Rc,
};
type Ptr = *mut c_void;
const CUDNN_RNN_DOUBLE_BIAS: i32 = 2;
const CUDNN_FMA_MATH: i32 = 3;
type Create = unsafe extern "C" fn(*mut Ptr) -> i32;
type Destroy = unsafe extern "C" fn(Ptr) -> i32;
type SetRnn = unsafe extern "C" fn(
    Ptr,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    Ptr,
    u32,
) -> i32;
type WeightSize = unsafe extern "C" fn(Ptr, Ptr, *mut usize) -> i32;
type WeightParams = unsafe extern "C" fn(
    Ptr,
    Ptr,
    i32,
    usize,
    *const c_void,
    i32,
    Ptr,
    *mut Ptr,
    Ptr,
    *mut Ptr,
) -> i32;
type SetData = unsafe extern "C" fn(Ptr, i32, i32, i32, i32, i32, *const i32, *const c_void) -> i32;
type TempSize = unsafe extern "C" fn(Ptr, Ptr, i32, Ptr, *mut usize, *mut usize) -> i32;
type Forward = unsafe extern "C" fn(
    Ptr,
    Ptr,
    i32,
    *const i32,
    Ptr,
    *const c_void,
    Ptr,
    Ptr,
    Ptr,
    *const c_void,
    Ptr,
    Ptr,
    *const c_void,
    Ptr,
    usize,
    *const c_void,
    usize,
    Ptr,
    usize,
    Ptr,
) -> i32;

pub struct Dnn {
    _library: Library,
    pub device: Rc<Device>,
    handle: Ptr,
    destroy: Destroy,
    error: unsafe extern "C" fn(i32) -> *const c_char,
    create_rnn: Create,
    destroy_rnn: Destroy,
    set_rnn: SetRnn,
    create_dropout: Create,
    destroy_dropout: Destroy,
    set_dropout: unsafe extern "C" fn(Ptr, Ptr, f32, Ptr, usize, u64) -> i32,
    create_tensor: Create,
    destroy_tensor: Destroy,
    set_tensor: unsafe extern "C" fn(Ptr, i32, i32, *const i32, *const i32) -> i32,
    get_tensor: unsafe extern "C" fn(Ptr, i32, *mut i32, *mut i32, *mut i32, *mut i32) -> i32,
    create_data: Create,
    destroy_data: Destroy,
    set_data: SetData,
    weight_size: WeightSize,
    weight_params: WeightParams,
    temp_size: TempSize,
    forward: Forward,
}
impl Dnn {
    pub fn load(device: Rc<Device>) -> Result<Rc<Self>> {
        let path = std::env::var_os("GLADOS_CUDNN_LIBRARY").unwrap_or_else(|| {
            if cfg!(windows) {
                "cudnn64_9.dll".into()
            } else {
                "libcudnn.so.9".into()
            }
        });
        // SAFETY: selected library must implement the documented cuDNN 9 C ABI.
        let library = unsafe { Library::new(path) }
            .context("load cuDNN; set GLADOS_CUDNN_LIBRARY and its dependency search path")?;
        macro_rules! symbol {
            ($name:literal,$ty:ty) => {{
                // SAFETY: signature matches the named documented C entry point.
                *unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }?
            }};
        }
        let create = symbol!("cudnnCreate", Create);
        let set_stream = symbol!("cudnnSetStream", unsafe extern "C" fn(Ptr, Ptr) -> i32);
        let mut api = Self {
            handle: ptr::null_mut(),
            device,
            destroy: symbol!("cudnnDestroy", Destroy),
            error: symbol!(
                "cudnnGetErrorString",
                unsafe extern "C" fn(i32) -> *const c_char
            ),
            create_rnn: symbol!("cudnnCreateRNNDescriptor", Create),
            destroy_rnn: symbol!("cudnnDestroyRNNDescriptor", Destroy),
            set_rnn: symbol!("cudnnSetRNNDescriptor_v8", SetRnn),
            create_dropout: symbol!("cudnnCreateDropoutDescriptor", Create),
            destroy_dropout: symbol!("cudnnDestroyDropoutDescriptor", Destroy),
            set_dropout: symbol!(
                "cudnnSetDropoutDescriptor",
                unsafe extern "C" fn(Ptr, Ptr, f32, Ptr, usize, u64) -> i32
            ),
            create_tensor: symbol!("cudnnCreateTensorDescriptor", Create),
            destroy_tensor: symbol!("cudnnDestroyTensorDescriptor", Destroy),
            set_tensor: symbol!(
                "cudnnSetTensorNdDescriptor",
                unsafe extern "C" fn(Ptr, i32, i32, *const i32, *const i32) -> i32
            ),
            get_tensor: symbol!(
                "cudnnGetTensorNdDescriptor",
                unsafe extern "C" fn(Ptr, i32, *mut i32, *mut i32, *mut i32, *mut i32) -> i32
            ),
            create_data: symbol!("cudnnCreateRNNDataDescriptor", Create),
            destroy_data: symbol!("cudnnDestroyRNNDataDescriptor", Destroy),
            set_data: symbol!("cudnnSetRNNDataDescriptor", SetData),
            weight_size: symbol!("cudnnGetRNNWeightSpaceSize", WeightSize),
            weight_params: symbol!("cudnnGetRNNWeightParams", WeightParams),
            temp_size: symbol!("cudnnGetRNNTempSpaceSizes", TempSize),
            forward: symbol!("cudnnRNNForward", Forward),
            _library: library,
        };
        // SAFETY: output pointer and stream belong to live objects.
        let status = unsafe { create(&mut api.handle) };
        api.check(status)?;
        api.check(unsafe { set_stream(api.handle, api.device.stream()) })?;
        Ok(Rc::new(api))
    }
    fn check(&self, status: i32) -> Result<()> {
        // SAFETY: cuDNN returns a static terminated error description.
        ensure!(
            status == 0,
            "cuDNN: {}",
            unsafe { CStr::from_ptr((self.error)(status)) }.to_string_lossy()
        );
        Ok(())
    }
    fn descriptor(self: &Rc<Self>, create: Create, destroy: Destroy) -> Result<Descriptor> {
        let mut raw = ptr::null_mut();
        // SAFETY: valid output location; returned descriptor owns this reference.
        self.check(unsafe { create(&mut raw) })?;
        ensure!(!raw.is_null(), "cuDNN returned a null descriptor");
        Ok(Descriptor {
            raw,
            destroy,
            _api: self.clone(),
        })
    }
    fn tensor_count(&self, desc: Ptr) -> Result<usize> {
        let (mut dtype, mut nd) = (0, 0);
        let mut dims = [0i32; 8];
        let mut strides = [0i32; 8];
        // SAFETY: arrays have the requested capacity; descriptor is live.
        self.check(unsafe {
            (self.get_tensor)(
                desc,
                8,
                &mut dtype,
                &mut nd,
                dims.as_mut_ptr(),
                strides.as_mut_ptr(),
            )
        })?;
        ensure!(
            dtype == 0 && (1..=8).contains(&nd),
            "unexpected cuDNN parameter descriptor: dtype={dtype}, rank={nd}"
        );
        let mut count = 1usize;
        for (&dimension, &stride) in dims[..nd as usize].iter().zip(&strides).rev() {
            ensure!(
                dimension > 0 && stride as usize == count,
                "noncontiguous cuDNN weight layout"
            );
            count = count
                .checked_mul(dimension as usize)
                .context("cuDNN shape overflow")?;
        }
        Ok(count)
    }
}
impl Drop for Dnn {
    fn drop(&mut self) {
        let _ = self.device.sync();
        if !self.handle.is_null() {
            // SAFETY: handle belongs to this object, library remains loaded.
            unsafe { (self.destroy)(self.handle) };
        }
    }
}
struct Descriptor {
    raw: Ptr,
    destroy: Destroy,
    _api: Rc<Dnn>,
}
impl Drop for Descriptor {
    fn drop(&mut self) {
        // SAFETY: unique descriptor ownership; retained API keeps library alive.
        unsafe { (self.destroy)(self.raw) };
    }
}

pub struct Rnn {
    api: Rc<Dnn>,
    desc: Descriptor,
    _dropout: Descriptor,
    weights: Buffer,
    bytes: usize,
    pub input: usize,
    pub hidden: usize,
    lstm: bool,
}
impl Rnn {
    pub fn load(weights: &Weights, api: Rc<Dnn>, prefix: &str, lstm: bool) -> Result<Self> {
        let shape = weights.shape(&format!("{prefix}.weight_ih_l0"))?;
        let gates = if lstm { 4 } else { 3 };
        ensure!(
            shape.len() == 2 && shape[0] % gates == 0,
            "RNN weights shape"
        );
        let input = shape[1];
        let hidden = shape[0] / gates;
        ensure!(
            input > 0 && hidden > 0 && input <= i32::MAX as usize && hidden <= i32::MAX as usize,
            "RNN dimension overflow"
        );
        let dropout = api.descriptor(api.create_dropout, api.destroy_dropout)?;
        // SAFETY: dropout disabled; no RNG state buffer needed.
        api.check(unsafe {
            (api.set_dropout)(dropout.raw, api.handle, 0., ptr::null_mut(), 0, 0)
        })?;
        let desc = api.descriptor(api.create_rnn, api.destroy_rnn)?;
        #[cfg(not(feature = "experimental-rnn"))]
        let algorithm = 0;
        // Persistent variants are faster but failed full-waveform parity. Keep
        // them available only in explicit development experiment builds.
        #[cfg(feature = "experimental-rnn")]
        let algorithm = match std::env::var("GLADOS_RNN_ALGORITHM").as_deref() {
            Ok("persistent") => 1,
            Ok("small") => 3,
            Ok("standard") | Err(std::env::VarError::NotPresent) => 0,
            Ok("postnet") => {
                if prefix == "acoustic.postnet.rnn" {
                    1
                } else {
                    0
                }
            }
            Ok("lstm") => {
                if lstm {
                    1
                } else {
                    0
                }
            }
            _ => anyhow::bail!(
                "GLADOS_RNN_ALGORITHM must be standard, persistent, small, postnet or lstm"
            ),
        };
        // LSTM/GRU, double bias, bidirectional, linear input, F32 storage/math,
        // no reduced-precision matrix multiplication, one layer, padded I/O.
        api.check(unsafe {
            (api.set_rnn)(
                desc.raw,
                algorithm,
                if lstm { 2 } else { 3 },
                CUDNN_RNN_DOUBLE_BIAS,
                1,
                0,
                0,
                0,
                CUDNN_FMA_MATH,
                input as i32,
                hidden as i32,
                hidden as i32,
                1,
                dropout.raw,
                1,
            )
        })?;
        let mut bytes = 0;
        api.check(unsafe { (api.weight_size)(api.handle, desc.raw, &mut bytes) })?;
        let packed = api.device.allocate(bytes.div_ceil(4))?;
        packed.zero()?;
        let mdesc = api.descriptor(api.create_tensor, api.destroy_tensor)?;
        let bdesc = api.descriptor(api.create_tensor, api.destroy_tensor)?;
        for direction in 0..2 {
            let suffix = if direction == 0 { "" } else { "_reverse" };
            for transform in 0..2 {
                let kind = if transform == 0 { "ih" } else { "hh" };
                let columns = if transform == 0 { input } else { hidden };
                let matrix = weights.f32(
                    &format!("{prefix}.weight_{kind}_l0{suffix}"),
                    &[gates * hidden, columns],
                )?;
                let bias = weights.f32(
                    &format!("{prefix}.bias_{kind}_l0{suffix}"),
                    &[gates * hidden],
                )?;
                for gate in 0..gates {
                    let (mut maddr, mut baddr) = (ptr::null_mut(), ptr::null_mut());
                    api.check(unsafe {
                        (api.weight_params)(
                            api.handle,
                            desc.raw,
                            direction,
                            bytes,
                            packed.pointer(),
                            (transform * gates + gate) as i32,
                            mdesc.raw,
                            &mut maddr,
                            bdesc.raw,
                            &mut baddr,
                        )
                    })?;
                    ensure!(
                        api.tensor_count(mdesc.raw)? == hidden * columns
                            && api.tensor_count(bdesc.raw)? == hidden,
                        "cuDNN parameter size mismatch"
                    );
                    let mo = (maddr as usize)
                        .checked_sub(packed.pointer() as usize)
                        .context("cuDNN matrix outside allocation")?;
                    let bo = (baddr as usize)
                        .checked_sub(packed.pointer() as usize)
                        .context("cuDNN bias outside allocation")?;
                    packed.upload_at(
                        mo,
                        &matrix[gate * hidden * columns * 4..(gate + 1) * hidden * columns * 4],
                    )?;
                    packed.upload_at(bo, &bias[gate * hidden * 4..(gate + 1) * hidden * 4])?;
                }
            }
        }
        Ok(Self {
            api,
            desc,
            _dropout: dropout,
            weights: packed,
            bytes,
            input,
            hidden,
            lstm,
        })
    }
    /// Time-major [T,I] input and [T,2H] output for batch one.
    pub fn run(&self, x: &Buffer, time: usize) -> Result<Buffer> {
        ensure!(
            time > 0 && time <= 65535 && x.len == time * self.input,
            "RNN input dimensions"
        );
        let xd = self
            .api
            .descriptor(self.api.create_data, self.api.destroy_data)?;
        let yd = self
            .api
            .descriptor(self.api.create_data, self.api.destroy_data)?;
        let hd = self
            .api
            .descriptor(self.api.create_tensor, self.api.destroy_tensor)?;
        let time = time as i32;
        let zero = 0f32;
        self.api.check(unsafe {
            (self.api.set_data)(
                xd.raw,
                0,
                0,
                time,
                1,
                self.input as i32,
                &time,
                (&zero as *const f32).cast(),
            )
        })?;
        self.api.check(unsafe {
            (self.api.set_data)(
                yd.raw,
                0,
                0,
                time,
                1,
                (2 * self.hidden) as i32,
                &time,
                (&zero as *const f32).cast(),
            )
        })?;
        let dims = [2, 1, self.hidden as i32];
        let strides = [self.hidden as i32, self.hidden as i32, 1];
        self.api.check(unsafe {
            (self.api.set_tensor)(hd.raw, 0, 3, dims.as_ptr(), strides.as_ptr())
        })?;
        let (mut work_bytes, mut reserve) = (0, 0);
        self.api.check(unsafe {
            (self.api.temp_size)(
                self.api.handle,
                self.desc.raw,
                0,
                xd.raw,
                &mut work_bytes,
                &mut reserve,
            )
        })?;
        ensure!(
            reserve == 0,
            "inference unexpectedly requested training state"
        );
        let workspace = self.api.device.allocate(work_bytes.div_ceil(4).max(1))?;
        let y = self.api.device.allocate(time as usize * 2 * self.hidden)?;
        self.api.check(unsafe {
            (self.api.forward)(
                self.api.handle,
                self.desc.raw,
                0,
                ptr::null(),
                xd.raw,
                x.pointer(),
                yd.raw,
                y.pointer(),
                hd.raw,
                ptr::null(),
                ptr::null_mut(),
                if self.lstm { hd.raw } else { ptr::null_mut() },
                ptr::null(),
                ptr::null_mut(),
                self.bytes,
                self.weights.pointer(),
                work_bytes,
                workspace.pointer(),
                0,
                ptr::null_mut(),
            )
        })?;
        Ok(y)
    }
}
