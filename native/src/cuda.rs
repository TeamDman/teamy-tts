use anyhow::{Result, anyhow, ensure};
use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
    rc::Rc,
};

unsafe extern "C" {
    fn glados_error() -> *const c_char;
    fn glados_create(out: *mut *mut c_void) -> i32;
    fn glados_destroy(session: *mut c_void);
    fn glados_alloc(session: *mut c_void, bytes: usize, out: *mut *mut c_void) -> i32;
    fn glados_free(session: *mut c_void, data: *mut c_void);
    fn glados_upload(
        session: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> i32;
    fn glados_download(
        session: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> i32;
    fn glados_sync(session: *mut c_void) -> i32;
    fn glados_stream(session: *mut c_void) -> *mut c_void;
    fn glados_zero(session: *mut c_void, data: *mut c_void, bytes: usize) -> i32;
    fn glados_copy(s: *mut c_void, dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    fn glados_transpose(
        s: *mut c_void,
        x: *const c_void,
        y: *mut c_void,
        rows: i32,
        cols: i32,
    ) -> i32;
    fn glados_embedding(
        s: *mut c_void,
        w: *const c_void,
        ids: *const c_void,
        y: *mut c_void,
        time: i32,
        width: i32,
    ) -> i32;
    fn glados_broadcast(
        s: *mut c_void,
        x: *const c_void,
        y: *mut c_void,
        time: i32,
        channels: i32,
    ) -> i32;
    fn glados_bn(
        s: *mut c_void,
        x: *mut c_void,
        scale: *const c_void,
        bias: *const c_void,
        time: i32,
        n: usize,
        relu: i32,
    ) -> i32;
    fn glados_pool(s: *mut c_void, x: *const c_void, y: *mut c_void, time: i32, n: usize) -> i32;
    fn glados_highway(
        s: *mut c_void,
        x: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        n: usize,
    ) -> i32;
    fn glados_add(
        s: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        n: usize,
    ) -> i32;
    fn glados_argmax(
        s: *mut c_void,
        x: *const c_void,
        y: *mut c_void,
        time: i32,
        channels: i32,
    ) -> i32;
    fn glados_gather(
        s: *mut c_void,
        x: *const c_void,
        ids: *const c_void,
        y: *mut c_void,
        input_time: i32,
        output_time: i32,
        channels: i32,
    ) -> i32;
    fn glados_phoneme_embed(
        s: *mut c_void,
        w: *const c_void,
        ids: *const c_void,
        pos: *const c_void,
        scale: f32,
        y: *mut c_void,
        time: i32,
    ) -> i32;
    fn glados_attention(
        s: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        scores: *mut c_void,
        y: *mut c_void,
        time: i32,
    ) -> i32;
    fn glados_layernorm(
        s: *mut c_void,
        x: *const c_void,
        residual: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        time: i32,
    ) -> i32;
    fn glados_conv(
        session: *mut c_void,
        x: *const c_void,
        w: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        col: *mut c_void,
        y: *mut c_void,
        input: i32,
        output: i32,
        time: i32,
        kernel: i32,
        dilation: i32,
        padding: i32,
        slope: f32,
    ) -> i32;
    fn glados_up(
        session: *mut c_void,
        x: *const c_void,
        w: *const c_void,
        bias: *const c_void,
        col: *mut c_void,
        phases: *mut c_void,
        y: *mut c_void,
        input: i32,
        output: i32,
        time: i32,
        stride: i32,
    ) -> i32;
    fn glados_mean3(
        session: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        out: *mut c_void,
        n: usize,
    ) -> i32;
    fn glados_tanh(session: *mut c_void, x: *mut c_void, n: usize) -> i32;
}

fn checked(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        // SAFETY: FFI returns a NUL-terminated thread-local error string.
        Err(anyhow!(
            unsafe { CStr::from_ptr(glados_error()) }
                .to_string_lossy()
                .into_owned()
        ))
    }
}

#[derive(Debug)]
pub struct Device {
    raw: NonNull<c_void>,
}
impl Device {
    pub(crate) fn stream(&self) -> *mut c_void {
        // SAFETY: live session; the stream remains owned by the session.
        unsafe { glados_stream(self.raw.as_ptr()) }
    }
    pub fn new() -> Result<Rc<Self>> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out parameter, the C function owns failure cleanup.
        checked(unsafe { glados_create(&mut raw) })?;
        Ok(Rc::new(Self {
            raw: NonNull::new(raw).ok_or_else(|| anyhow!("null CUDA session"))?,
        }))
    }
    pub fn allocate(self: &Rc<Self>, n: usize) -> Result<Buffer> {
        ensure!(n > 0, "empty CUDA allocation");
        let bytes = n
            .checked_mul(4)
            .ok_or_else(|| anyhow!("allocation overflow"))?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid session and output pointer; allocation is owned below.
        checked(unsafe { glados_alloc(self.raw.as_ptr(), bytes, &mut raw) })?;
        Ok(Buffer {
            device: self.clone(),
            raw: NonNull::new(raw).ok_or_else(|| anyhow!("null CUDA allocation"))?,
            len: n,
        })
    }
    pub fn upload(self: &Rc<Self>, bytes: &[u8]) -> Result<Buffer> {
        ensure!(bytes.len() % 4 == 0, "F32 byte alignment");
        let buffer = self.allocate(bytes.len() / 4)?;
        // SAFETY: allocation matches the slice size; upload synchronizes before return.
        checked(unsafe {
            glados_upload(
                self.raw.as_ptr(),
                buffer.raw.as_ptr(),
                bytes.as_ptr().cast(),
                bytes.len(),
            )
        })?;
        Ok(buffer)
    }
    pub fn upload_f32(self: &Rc<Self>, values: &[f32]) -> Result<Buffer> {
        let buffer = self.allocate(values.len())?;
        // SAFETY: live contiguous F32 slice and exact-size device allocation.
        checked(unsafe {
            glados_upload(
                self.raw.as_ptr(),
                buffer.raw.as_ptr(),
                values.as_ptr().cast(),
                values.len() * 4,
            )
        })?;
        Ok(buffer)
    }
    pub fn sync(&self) -> Result<()> {
        // SAFETY: session remains alive through this call.
        checked(unsafe { glados_sync(self.raw.as_ptr()) })
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: unique session owner; buffers retain an Rc until they are freed.
        unsafe { glados_destroy(self.raw.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct Buffer {
    device: Rc<Device>,
    raw: NonNull<c_void>,
    pub len: usize,
}
impl Buffer {
    pub(crate) fn pointer(&self) -> *mut c_void {
        self.raw.as_ptr()
    }
    pub(crate) fn zero(&self) -> Result<()> {
        // SAFETY: live allocation, exact byte extent.
        checked(unsafe { glados_zero(self.device.raw.as_ptr(), self.raw.as_ptr(), self.len * 4) })
    }
    pub(crate) fn upload_at(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        ensure!(
            offset
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.len * 4),
            "upload outside allocation"
        );
        // SAFETY: validated byte range inside allocation; host slice lives until synchronized upload returns.
        checked(unsafe {
            glados_upload(
                self.device.raw.as_ptr(),
                self.raw.as_ptr().cast::<u8>().add(offset).cast(),
                bytes.as_ptr().cast(),
                bytes.len(),
            )
        })
    }
    pub fn download(&self) -> Result<Vec<f32>> {
        let mut values = vec![0.; self.len];
        // SAFETY: exact-sized initialized output; download synchronizes before return.
        checked(unsafe {
            glados_download(
                self.device.raw.as_ptr(),
                values.as_mut_ptr().cast(),
                self.raw.as_ptr(),
                self.len * 4,
            )
        })?;
        Ok(values)
    }
    pub fn download_i32(&self) -> Result<Vec<i32>> {
        let mut values = vec![0i32; self.len];
        checked(unsafe {
            glados_download(
                self.device.raw.as_ptr(),
                values.as_mut_ptr().cast(),
                self.pointer(),
                self.len * 4,
            )
        })?;
        Ok(values)
    }
    pub fn phoneme_embedding(
        &self,
        ids: &Buffer,
        pos: &Buffer,
        scale: f32,
        time: usize,
    ) -> Result<Self> {
        ensure!(
            time <= 5000 && ids.len == time && self.len == 64 * 512 && pos.len == 5000 * 512,
            "phonemizer embedding shape"
        );
        let out = self.device.allocate(time * 512)?;
        checked(unsafe {
            glados_phoneme_embed(
                self.device.raw.as_ptr(),
                self.pointer(),
                ids.pointer(),
                pos.pointer(),
                scale,
                out.pointer(),
                time.try_into()?,
            )
        })?;
        Ok(out)
    }
    pub fn attention(&self, k: &Buffer, v: &Buffer, time: usize) -> Result<Self> {
        ensure!(
            time > 0
                && time <= 5000
                && self.len == time * 512
                && self.len == k.len
                && self.len == v.len,
            "attention shape"
        );
        let scores = self.device.allocate(4 * time * time)?;
        let out = self.device.allocate(self.len)?;
        checked(unsafe {
            glados_attention(
                self.device.raw.as_ptr(),
                self.pointer(),
                k.pointer(),
                v.pointer(),
                scores.pointer(),
                out.pointer(),
                time.try_into()?,
            )
        })?;
        Ok(out)
    }
    pub fn layer_norm(
        &self,
        residual: Option<&Buffer>,
        gamma: &Buffer,
        beta: &Buffer,
        time: usize,
    ) -> Result<Self> {
        ensure!(
            time > 0
                && self.len == time * 512
                && gamma.len == 512
                && beta.len == 512
                && residual.is_none_or(|r| r.len == self.len),
            "layer norm shape"
        );
        let out = self.device.allocate(self.len)?;
        checked(unsafe {
            glados_layernorm(
                self.device.raw.as_ptr(),
                self.pointer(),
                residual.map_or(std::ptr::null(), |r| r.pointer()),
                gamma.pointer(),
                beta.pointer(),
                out.pointer(),
                time.try_into()?,
            )
        })?;
        Ok(out)
    }
    pub fn conv(
        &self,
        w: &Buffer,
        b: &Buffer,
        residual: Option<&Buffer>,
        input: usize,
        output: usize,
        time: usize,
        kernel: usize,
        dilation: usize,
        slope: f32,
    ) -> Result<Buffer> {
        ensure!(
            self.len == input * time && w.len == output * input * kernel && b.len == output,
            "convolution dimensions"
        );
        ensure!(
            residual.is_none_or(|r| r.len == output * time),
            "residual dimensions"
        );
        ensure!(
            input * kernel <= i32::MAX as usize
                && output <= i32::MAX as usize
                && time <= i32::MAX as usize,
            "convolution dimension overflow"
        );
        let col = if kernel == 1 && slope == 1.0 {
            None
        } else {
            Some(self.device.allocate(input * kernel * time)?)
        };
        let y = self.device.allocate(output * time)?;
        // CBHG uses padding k/2 even for even kernels, then trims to input time.
        let padding = (kernel / 2) * dilation;
        // SAFETY: checked dimensions match all allocations; one session stream orders lifetimes.
        checked(unsafe {
            glados_conv(
                self.device.raw.as_ptr(),
                self.raw.as_ptr(),
                w.raw.as_ptr(),
                b.raw.as_ptr(),
                residual.map_or(std::ptr::null(), |r| r.raw.as_ptr()),
                col.as_ref()
                    .map_or(std::ptr::null_mut(), |c| c.raw.as_ptr()),
                y.raw.as_ptr(),
                input as i32,
                output as i32,
                time as i32,
                kernel as i32,
                dilation as i32,
                padding as i32,
                slope,
            )
        })?;
        Ok(y)
    }
    pub fn up(
        &self,
        w: &Buffer,
        b: &Buffer,
        input: usize,
        output: usize,
        time: usize,
        stride: usize,
    ) -> Result<Buffer> {
        ensure!(
            self.len == input * time && w.len == stride * output * 2 * input && b.len == output,
            "upsampling dimensions"
        );
        ensure!(
            2 * input <= i32::MAX as usize
                && output <= i32::MAX as usize
                && time * stride <= i32::MAX as usize,
            "upsampling dimension overflow"
        );
        let col = self.device.allocate(stride * 2 * input * time)?;
        let phases = self.device.allocate(stride * output * time)?;
        let y = self.device.allocate(output * time * stride)?;
        // SAFETY: dimensions checked; packed weights and scratch buffers have exact extents.
        checked(unsafe {
            glados_up(
                self.device.raw.as_ptr(),
                self.raw.as_ptr(),
                w.raw.as_ptr(),
                b.raw.as_ptr(),
                col.raw.as_ptr(),
                phases.raw.as_ptr(),
                y.raw.as_ptr(),
                input as i32,
                output as i32,
                time as i32,
                stride as i32,
            )
        })?;
        Ok(y)
    }
    pub fn mean3(&self, b: &Buffer, c: &Buffer) -> Result<Buffer> {
        ensure!(self.len == b.len && self.len == c.len, "mean dimensions");
        let y = self.device.allocate(self.len)?;
        // SAFETY: equal-size buffers, all created in the same session.
        checked(unsafe {
            glados_mean3(
                self.device.raw.as_ptr(),
                self.raw.as_ptr(),
                b.raw.as_ptr(),
                c.raw.as_ptr(),
                y.raw.as_ptr(),
                self.len,
            )
        })?;
        Ok(y)
    }
    pub fn tanh(self) -> Result<Self> {
        // SAFETY: uniquely owned output buffer, in-place kernel stays in bounds.
        checked(unsafe { glados_tanh(self.device.raw.as_ptr(), self.raw.as_ptr(), self.len) })?;
        Ok(self)
    }
    pub fn transpose(&self, rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            rows * cols == self.len && rows > 0 && cols > 0,
            "transpose dimensions"
        );
        let out = self.device.allocate(self.len)?;
        // SAFETY: source/destination extents match the checked matrix dimensions.
        checked(unsafe {
            glados_transpose(
                self.device.raw.as_ptr(),
                self.pointer(),
                out.pointer(),
                rows.try_into()?,
                cols.try_into()?,
            )
        })?;
        Ok(out)
    }
    /// IDs are either validated host token IDs or the result of bounded argmax.
    pub fn embedding(&self, ids: &Buffer, time: usize, width: usize) -> Result<Self> {
        ensure!(
            ids.len == time && width > 0 && self.len % width == 0,
            "embedding dimensions"
        );
        let out = self.device.allocate(time * width)?;
        checked(unsafe {
            glados_embedding(
                self.device.raw.as_ptr(),
                self.pointer(),
                ids.pointer(),
                out.pointer(),
                time.try_into()?,
                width.try_into()?,
            )
        })?;
        Ok(out)
    }
    pub fn broadcast(&self, time: usize) -> Result<Self> {
        let out = self.device.allocate(time * self.len)?;
        checked(unsafe {
            glados_broadcast(
                self.device.raw.as_ptr(),
                self.pointer(),
                out.pointer(),
                time.try_into()?,
                self.len.try_into()?,
            )
        })?;
        Ok(out)
    }
    pub fn concat(parts: &[&Buffer]) -> Result<Self> {
        ensure!(!parts.is_empty(), "empty concatenation");
        let device = &parts[0].device;
        ensure!(
            parts.iter().all(|p| Rc::ptr_eq(device, &p.device)),
            "mixed CUDA sessions"
        );
        let out = device.allocate(parts.iter().map(|p| p.len).sum())?;
        let mut offset = 0;
        for part in parts {
            // SAFETY: concatenated output allocated to sum of source lengths.
            checked(unsafe {
                glados_copy(
                    device.raw.as_ptr(),
                    out.pointer().cast::<u8>().add(offset).cast(),
                    part.pointer(),
                    part.len * 4,
                )
            })?;
            offset += part.len * 4;
        }
        Ok(out)
    }
    pub fn batch_norm(
        self,
        scale: &Buffer,
        bias: &Buffer,
        time: usize,
        relu: bool,
    ) -> Result<Self> {
        ensure!(
            time > 0
                && self.len % time == 0
                && scale.len == self.len / time
                && bias.len == scale.len,
            "batch norm dimensions"
        );
        checked(unsafe {
            glados_bn(
                self.device.raw.as_ptr(),
                self.pointer(),
                scale.pointer(),
                bias.pointer(),
                time.try_into()?,
                self.len,
                i32::from(relu),
            )
        })?;
        Ok(self)
    }
    pub fn max_pool(&self, time: usize) -> Result<Self> {
        ensure!(time > 0 && self.len % time == 0, "pool dimensions");
        let out = self.device.allocate(self.len)?;
        checked(unsafe {
            glados_pool(
                self.device.raw.as_ptr(),
                self.pointer(),
                out.pointer(),
                time.try_into()?,
                self.len,
            )
        })?;
        Ok(out)
    }
    pub fn highway(&self, a: &Buffer, b: &Buffer) -> Result<Self> {
        ensure!(self.len == a.len && self.len == b.len, "highway dimensions");
        let out = self.device.allocate(self.len)?;
        checked(unsafe {
            glados_highway(
                self.device.raw.as_ptr(),
                self.pointer(),
                a.pointer(),
                b.pointer(),
                out.pointer(),
                self.len,
            )
        })?;
        Ok(out)
    }
    pub fn add(&self, b: &Buffer) -> Result<Self> {
        ensure!(self.len == b.len, "add dimensions");
        let out = self.device.allocate(self.len)?;
        checked(unsafe {
            glados_add(
                self.device.raw.as_ptr(),
                self.pointer(),
                b.pointer(),
                out.pointer(),
                self.len,
            )
        })?;
        Ok(out)
    }
    pub fn argmax(&self, time: usize, channels: usize) -> Result<Self> {
        ensure!(
            time > 0 && channels > 0 && self.len == time * channels,
            "argmax dimensions"
        );
        let out = self.device.allocate(time)?;
        checked(unsafe {
            glados_argmax(
                self.device.raw.as_ptr(),
                self.pointer(),
                out.pointer(),
                time.try_into()?,
                channels.try_into()?,
            )
        })?;
        Ok(out)
    }
    pub fn gather(&self, indices: &[i32], time: usize, channels: usize) -> Result<Self> {
        ensure!(
            self.len == time * channels && indices.iter().all(|&i| i >= 0 && (i as usize) < time),
            "gather dimensions/indices"
        );
        let bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ids = self.device.upload(&bytes)?;
        let out = self.device.allocate(indices.len() * channels)?;
        checked(unsafe {
            glados_gather(
                self.device.raw.as_ptr(),
                self.pointer(),
                ids.pointer(),
                out.pointer(),
                time.try_into()?,
                indices.len().try_into()?,
                channels.try_into()?,
            )
        })?;
        Ok(out)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: unique allocation owner; async free follows all uses on the same stream.
        unsafe { glados_free(self.device.raw.as_ptr(), self.raw.as_ptr()) };
    }
}
