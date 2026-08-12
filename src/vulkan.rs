//! Vulkan capability discovery and model-kernel execution for the experimental
//! backend.
//!
//! The current supported boundary is the prepared embedding, fixed-shape
//! acoustic continuation/postnet, and the complete `GLaDOS` HiFi-GAN vocoder.
//! The predictor/prenet prefix remains behind the Burn reference boundary
//! while the Vulkan device-resident batches are measured and extended.

use ash::Entry;
use ash::vk;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use std::ffi::CString;

const VECTOR_ADD_ELEMENT_COUNT: usize = 1024;
const VECTOR_ADD_WORKGROUP_COUNT: u32 = 4;
const VECTOR_ADD_BUFFER_SIZE: vk::DeviceSize =
    (VECTOR_ADD_ELEMENT_COUNT * std::mem::size_of::<f32>()) as vk::DeviceSize;
const MATMUL_DIMENSION: usize = 16;
const MATMUL_ELEMENT_COUNT: usize = MATMUL_DIMENSION * MATMUL_DIMENSION;
const MATMUL_BUFFER_SIZE: vk::DeviceSize =
    (MATMUL_ELEMENT_COUNT * std::mem::size_of::<f32>()) as vk::DeviceSize;
static VECTOR_ADD_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_VECTOR_ADD_SPV"));
static MATMUL_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_MATMUL_SPV"));
static EMBEDDING_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_EMBEDDING_SPV"));
static CONV1D_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_CONV1D_SPV"));
static CONV_TRANSPOSE1D_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_CONV_TRANSPOSE1D_SPV"));
static ELEMENTWISE_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_ELEMENTWISE_SPV"));
static LINEAR_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_LINEAR_SPV"));
static LENGTH_REGULATE_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_LENGTH_REGULATE_SPV"));
static LSTM_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_LSTM_SPV"));
static BATCH_NORM_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_BATCH_NORM_SPV"));
static MAX_POOL1D_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_MAX_POOL1D_SPV"));
static GRU_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_GRU_SPV"));
static COPY_CHANNELS_SPIR_V: &[u8] = include_bytes!(env!("TEAMY_TTS_COPY_CHANNELS_SPV"));

/// Reusable Ash device state for the model-oriented Vulkan backend.
///
/// The probe below intentionally creates short-lived devices so it can report
/// independent capability facts. Model inference uses this context instead:
/// the instance, logical device, queue, and memory properties stay alive for
/// the loaded runtime.
pub struct VulkanContext {
    _entry: Entry,
    instance: ash::Instance,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
    timestamp_period_ns: f32,
    timestamp_valid_bits: u32,
    profile_batches: bool,
    embedding_pipeline: EmbeddingPipeline,
    conv1d_pipeline: Conv1dPipeline,
    conv_transpose1d_pipeline: ConvTranspose1dPipeline,
    elementwise_pipeline: ElementwisePipeline,
    linear_pipeline: LinearPipeline,
    length_regulate_pipeline: LengthRegulatePipeline,
    lstm_pipeline: LstmPipeline,
    batch_norm_pipeline: SimpleComputePipeline,
    max_pool1d_pipeline: SimpleComputePipeline,
    gru_pipeline: SimpleComputePipeline,
    copy_channels_pipeline: SimpleComputePipeline,
    embedding_weights: Option<PersistentEmbeddingWeights>,
    model_buffers: Vec<PersistentModelBuffer>,
}

struct EmbeddingPipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct Conv1dPipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct ConvTranspose1dPipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct ElementwisePipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct LinearPipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct LengthRegulatePipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct LstmPipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct SimpleComputePipeline {
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct PersistentEmbeddingWeights {
    buffer: BufferAllocation,
    vocabulary_size: usize,
    embedding_dimension: usize,
    byte_len: vk::DeviceSize,
}

struct PersistentModelBuffer {
    allocation: BufferAllocation,
    byte_len: vk::DeviceSize,
}

/// An index into model-weight storage owned by a [`VulkanContext`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct VulkanModelBufferHandle(usize);

/// An index into temporary storage owned by a recorded Vulkan batch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VulkanBatchBufferHandle(usize);

impl std::fmt::Debug for VulkanContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanContext")
            .field("device_name", &self.device_name)
            .field("queue_family_index", &self.queue_family_index)
            .finish_non_exhaustive()
    }
}

impl VulkanContext {
    /// Open the preferred compute-capable Vulkan device.
    ///
    /// # Errors
    ///
    /// Returns an actionable error when the Vulkan loader, instance, device, or
    /// compute queue cannot be created.
    ///
    /// # Panics
    ///
    /// Panics only if the static application or engine names unexpectedly
    /// contain an interior NUL.
    #[expect(
        clippy::too_many_lines,
        reason = "Vulkan context construction keeps device selection, pipeline creation, and ownership in one auditable boundary."
    )]
    pub fn new() -> Result<Self> {
        // SAFETY: Entry::load performs the platform Vulkan-loader lookup and
        // returns an owned entry whose function pointers remain valid while it
        // is alive.
        let entry = unsafe { Entry::load() }.wrap_err(
            "failed to load the Vulkan loader; install a Vulkan driver/runtime before using the Vulkan backend",
        )?;
        // SAFETY: Querying the loader version does not require an instance.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .wrap_err("failed to query the Vulkan loader version")?
            .unwrap_or(vk::API_VERSION_1_0)
            .min(vk::API_VERSION_1_3);
        let application_name =
            CString::new("teamy-tts").expect("static application name has no interior NUL");
        let engine_name =
            CString::new("teamy-tts-vulkan").expect("static engine name has no interior NUL");
        let application_info = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(loader_version);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&application_info);
        // SAFETY: The create-info points to data alive for the duration of the
        // call, and no optional extensions are required by the base backend.
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .wrap_err("failed to create the Vulkan inference instance")?;
        // SAFETY: The instance is alive and owns the returned physical-device
        // handles for the duration of this context construction.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .wrap_err("failed to enumerate Vulkan inference devices")?;
        let mut selected = None;
        let mut selected_report = None;
        for physical_device in physical_devices {
            let report = inspect_device(&entry, &instance, physical_device)?;
            let preferred = report.compute_queue_family_index.is_some()
                && (report.device_type == "DISCRETE_GPU" || report.name.contains("RTX 4090"));
            if report.compute_queue_family_index.is_some() && (selected.is_none() || preferred) {
                selected = Some(physical_device);
                selected_report = Some(report);
                if preferred {
                    break;
                }
            }
        }
        let physical_device =
            selected.ok_or_else(|| eyre::eyre!("no Vulkan physical device was enumerated"))?;
        let report = selected_report
            .ok_or_else(|| eyre::eyre!("Vulkan device selection produced no capability report"))?;
        let queue_family_index = report
            .compute_queue_family_index
            .ok_or_else(|| eyre::eyre!("selected Vulkan device has no compute queue"))?;
        // SAFETY: The physical device belongs to this live instance.
        let physical_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let device_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
        // SAFETY: The queue family was reported by this physical device and
        // the create-info remains alive for the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .wrap_err("failed to create the Vulkan inference device")?;
        // SAFETY: The queue family and queue index were validated above.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        // SAFETY: The physical device belongs to this live instance.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let embedding_pipeline = create_embedding_pipeline(&device)?;
        let conv1d_pipeline = create_conv1d_pipeline(&device)?;
        let conv_transpose1d_pipeline = create_conv_transpose1d_pipeline(&device)?;
        let elementwise_pipeline = create_elementwise_pipeline(&device)?;
        let linear_pipeline = create_linear_pipeline(&device)?;
        let length_regulate_pipeline = create_length_regulate_pipeline(&device)?;
        let lstm_pipeline = create_lstm_pipeline(&device)?;
        let batch_norm_pipeline = create_simple_compute_pipeline(
            &device,
            BATCH_NORM_SPIR_V,
            6,
            std::mem::size_of::<[u32; 3]>(),
            "batch normalization",
        )?;
        let max_pool1d_pipeline = create_simple_compute_pipeline(
            &device,
            MAX_POOL1D_SPIR_V,
            2,
            std::mem::size_of::<[u32; 6]>(),
            "MaxPool1d",
        )?;
        let gru_pipeline = create_simple_compute_pipeline(
            &device,
            GRU_SPIR_V,
            6,
            std::mem::size_of::<[u32; 5]>(),
            "GRU",
        )?;
        let copy_channels_pipeline = create_simple_compute_pipeline(
            &device,
            COPY_CHANNELS_SPIR_V,
            2,
            std::mem::size_of::<[u32; 3]>(),
            "channel copy",
        )?;
        Ok(Self {
            _entry: entry,
            instance,
            _physical_device: physical_device,
            device,
            queue,
            queue_family_index,
            memory_properties,
            device_name: report.name,
            timestamp_period_ns: physical_properties.limits.timestamp_period,
            timestamp_valid_bits: report.compute_timestamp_valid_bits.unwrap_or_default(),
            profile_batches: profile_batches_enabled(),
            embedding_pipeline,
            conv1d_pipeline,
            conv_transpose1d_pipeline,
            elementwise_pipeline,
            linear_pipeline,
            length_regulate_pipeline,
            lstm_pipeline,
            batch_norm_pipeline,
            max_pool1d_pipeline,
            gru_pipeline,
            copy_channels_pipeline,
            embedding_weights: None,
            model_buffers: Vec::new(),
        })
    }

    /// Return the selected physical-device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Upload the prepared acoustic embedding once for reuse by later
    /// synthesis calls.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied model shape or device-local upload is
    /// invalid. Calling this method again replaces the previous upload after
    /// waiting for the device to become idle.
    pub fn prepare_embedding(
        &mut self,
        weights: &[f32],
        vocabulary_size: usize,
        embedding_dimension: usize,
    ) -> Result<()> {
        let expected_weights = vocabulary_size
            .checked_mul(embedding_dimension)
            .ok_or_else(|| eyre::eyre!("Vulkan embedding dimensions overflow"))?;
        if weights.len() != expected_weights {
            bail!(
                "Vulkan embedding weights have {}, expected {}",
                weights.len(),
                expected_weights
            );
        }
        // SAFETY: Replacing a persistent model buffer is only allowed after
        // all previous dispatches have completed.
        unsafe { self.device.device_wait_idle() }
            .wrap_err("failed to idle Vulkan before replacing embedding weights")?;
        if let Some(previous) = self.embedding_weights.take() {
            destroy_buffer(&self.device, &previous.buffer);
        }
        let byte_len = u64::try_from(std::mem::size_of_val(weights))
            .wrap_err("Vulkan embedding weights are too large")?;
        let buffer = upload_device_local_buffer(
            &self.device,
            &self.memory_properties,
            self.queue,
            self.queue_family_index,
            weights_as_bytes(weights),
            "Vulkan embedding weights",
        )?;
        self.embedding_weights = Some(PersistentEmbeddingWeights {
            buffer,
            vocabulary_size,
            embedding_dimension,
            byte_len,
        });
        Ok(())
    }

    /// Dispatch the prepared acoustic embedding without re-uploading its
    /// weights.
    ///
    /// # Errors
    ///
    /// Returns an error when `prepare_embedding` has not been called, token
    /// IDs are outside the uploaded vocabulary, or dispatch/result transfer
    /// fails.
    pub fn dispatch_prepared_embedding(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let prepared = self
            .embedding_weights
            .as_ref()
            .ok_or_else(|| eyre::eyre!("Vulkan embedding weights have not been prepared"))?;
        self.dispatch_embedding(
            &[],
            prepared.vocabulary_size,
            prepared.embedding_dimension,
            tokens,
        )
    }

    /// Upload one reusable model tensor for a device-resident inference batch.
    ///
    /// Model weights are uploaded once through a host-visible staging buffer
    /// and retained in device-local storage; they are not recreated for every
    /// convolution dispatch.
    pub(crate) fn prepare_model_buffer(
        &mut self,
        values: &[f32],
    ) -> Result<VulkanModelBufferHandle> {
        if values.is_empty() {
            bail!("Vulkan model buffers cannot be empty");
        }
        let byte_len = u64::try_from(std::mem::size_of_val(values))
            .wrap_err("Vulkan model buffer is too large")?;
        let allocation = upload_device_local_buffer(
            &self.device,
            &self.memory_properties,
            self.queue,
            self.queue_family_index,
            weights_as_bytes(values),
            "Vulkan model buffer",
        )?;
        let index = self.model_buffers.len();
        self.model_buffers.push(PersistentModelBuffer {
            allocation,
            byte_len,
        });
        Ok(VulkanModelBufferHandle(index))
    }

    /// Dispatch one persistent float32 linear layer for a `[channels,
    /// sequence]` tensor and read its `[output_channels, sequence]` result.
    ///
    /// This boundary is intentionally synchronous for the first acoustic
    /// migration slice. Later stages can record this operation into the same
    /// device-resident batch as the vocoder once their upstream tensors also
    /// live in Vulkan buffers.
    ///
    /// # Errors
    ///
    /// Returns an error when the model handles, shapes, dispatch, or result
    /// transfer are invalid.
    #[expect(
        clippy::too_many_arguments,
        reason = "The linear model-kernel boundary exposes fixed-shape dimensions and persistent handles explicitly."
    )]
    pub(crate) fn dispatch_linear(
        &self,
        weights: VulkanModelBufferHandle,
        bias: VulkanModelBufferHandle,
        input: &[f32],
        input_channels: usize,
        sequence_length: usize,
        output_channels: usize,
        has_bias: bool,
    ) -> Result<Vec<f32>> {
        let input_count = input_channels
            .checked_mul(sequence_length)
            .ok_or_else(|| eyre::eyre!("Vulkan linear input dimensions overflow"))?;
        if input.len() != input_count {
            bail!(
                "Vulkan linear input has {}, expected {}",
                input.len(),
                input_count
            );
        }
        let output_count = output_channels
            .checked_mul(sequence_length)
            .ok_or_else(|| eyre::eyre!("Vulkan linear output dimensions overflow"))?;
        let mut batch = self.begin_batch()?;
        let input_buffer = batch.alloc_buffer(input_count)?;
        let output_buffer = batch.alloc_buffer(output_count)?;
        batch.write_buffer(input_buffer, input)?;
        batch.dispatch_linear(
            weights,
            bias,
            input_buffer,
            output_buffer,
            input_channels,
            sequence_length,
            output_channels,
            has_bias,
        )?;
        batch.finish(output_buffer, output_count)
    }

    /// Dispatch one fixed-shape length-regulation operation and read its
    /// channel-major `[channels, frames]` result.
    ///
    /// Durations use the same rounding rule as the Burn reference graph:
    /// `(duration + 0.5).max(0.0) as usize`. Frames beyond the sum of the
    /// rounded durations are explicitly zero-filled, matching padded Burn
    /// batches.
    ///
    /// This first migration boundary is intentionally synchronous. The
    /// operation can join the model batch once its conditioning input is also
    /// produced in Vulkan.
    pub(crate) fn dispatch_length_regulate(
        &self,
        input: &[f32],
        channels: usize,
        token_count: usize,
        durations: &[f32],
        frame_count: usize,
    ) -> Result<Vec<f32>> {
        if channels == 0 || token_count == 0 || frame_count == 0 {
            bail!("Vulkan length regulation dimensions must be non-zero");
        }
        if durations.len() != token_count {
            bail!(
                "Vulkan length regulation has {} durations, expected {}",
                durations.len(),
                token_count
            );
        }
        let input_count = channels
            .checked_mul(token_count)
            .ok_or_else(|| eyre::eyre!("Vulkan length regulation input dimensions overflow"))?;
        if input.len() != input_count {
            bail!(
                "Vulkan length regulation input has {}, expected {}",
                input.len(),
                input_count
            );
        }
        let output_count = channels
            .checked_mul(frame_count)
            .ok_or_else(|| eyre::eyre!("Vulkan length regulation output dimensions overflow"))?;
        let mut batch = self.begin_batch()?;
        let input_buffer = batch.alloc_buffer(input_count)?;
        let duration_buffer = batch.alloc_buffer(token_count)?;
        let output_buffer = batch.alloc_buffer(output_count)?;
        batch.write_buffer(input_buffer, input)?;
        batch.write_buffer(duration_buffer, durations)?;
        batch.dispatch_length_regulate(
            input_buffer,
            duration_buffer,
            output_buffer,
            channels,
            token_count,
            frame_count,
        )?;
        batch.finish(output_buffer, output_count)
    }

    /// Dispatch one batch-one fused LSTM direction. The fixed kernel uses a
    /// single 256-lane workgroup and keeps recurrent state in shared memory;
    /// this is a correctness/migration boundary until a sequence-specialized
    /// recurrent strategy proves faster on the target GPU.
    #[expect(
        clippy::too_many_arguments,
        reason = "The LSTM model-kernel boundary exposes persistent handles and fixed-shape dimensions explicitly."
    )]
    pub(crate) fn dispatch_lstm(
        &self,
        input_weights: VulkanModelBufferHandle,
        hidden_weights: VulkanModelBufferHandle,
        bias: VulkanModelBufferHandle,
        input: &[f32],
        input_channels: usize,
        hidden_channels: usize,
        sequence_length: usize,
        reverse: bool,
    ) -> Result<Vec<f32>> {
        if input_channels == 0 || hidden_channels == 0 || sequence_length == 0 {
            bail!("Vulkan LSTM dimensions must be non-zero");
        }
        if hidden_channels > 512 {
            bail!(
                "Vulkan LSTM hidden size {} exceeds the fixed kernel limit 512",
                hidden_channels
            );
        }
        let input_count = input_channels
            .checked_mul(sequence_length)
            .ok_or_else(|| eyre::eyre!("Vulkan LSTM input dimensions overflow"))?;
        if input.len() != input_count {
            bail!(
                "Vulkan LSTM input has {}, expected {}",
                input.len(),
                input_count
            );
        }
        let output_count = hidden_channels
            .checked_mul(sequence_length)
            .ok_or_else(|| eyre::eyre!("Vulkan LSTM output dimensions overflow"))?;
        let mut batch = self.begin_batch()?;
        let input_buffer = batch.alloc_buffer(input_count)?;
        let output_buffer = batch.alloc_buffer(output_count)?;
        batch.write_buffer(input_buffer, input)?;
        batch.dispatch_lstm(
            input_weights,
            hidden_weights,
            bias,
            input_buffer,
            output_buffer,
            input_channels,
            hidden_channels,
            sequence_length,
            reverse,
        )?;
        batch.finish(output_buffer, output_count)
    }

    /// Begin one recorded compute batch. Intermediate tensors remain on the
    /// device until `finish`, avoiding a host round trip and command/fence
    /// creation for every model layer.
    pub(crate) fn begin_batch(&self) -> Result<VulkanBatch<'_>> {
        VulkanBatch::new(self)
    }

    /// Run a batch-one, ungrouped float32 `ConvTranspose1d` through the cached
    /// Vulkan pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes, padding, buffer creation, dispatch, or
    /// result transfer is invalid.
    #[expect(
        clippy::too_many_arguments,
        reason = "The kernel boundary exposes the fixed-shape transposed-convolution parameters explicitly."
    )]
    pub fn dispatch_conv_transpose1d(
        &self,
        weights: &[f32],
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        bias: Option<&[f32]>,
        input: &[f32],
        input_length: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        padding_out: usize,
    ) -> Result<Vec<f32>> {
        if input_channels == 0 || output_channels == 0 || kernel_size == 0 {
            bail!("Vulkan ConvTranspose1d dimensions must be non-zero");
        }
        if input_length == 0 || stride == 0 || dilation == 0 {
            bail!("Vulkan ConvTranspose1d input length, stride, and dilation must be non-zero");
        }
        let expected_weights = input_channels
            .checked_mul(output_channels)
            .and_then(|value| value.checked_mul(kernel_size))
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d weight dimensions overflow"))?;
        if weights.len() != expected_weights {
            bail!(
                "Vulkan ConvTranspose1d weights have {}, expected {}",
                weights.len(),
                expected_weights
            );
        }
        let expected_input = input_channels
            .checked_mul(input_length)
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d input dimensions overflow"))?;
        if input.len() != expected_input {
            bail!(
                "Vulkan ConvTranspose1d input has {}, expected {}",
                input.len(),
                expected_input
            );
        }
        if let Some(bias) = bias
            && bias.len() != output_channels
        {
            bail!(
                "Vulkan ConvTranspose1d bias has {}, expected {}",
                bias.len(),
                output_channels
            );
        }
        let effective_kernel = dilation
            .checked_mul(kernel_size.saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d effective kernel overflows"))?;
        let untrimmed_length = input_length
            .saturating_sub(1)
            .checked_mul(stride)
            .and_then(|value| value.checked_add(effective_kernel))
            .and_then(|value| value.checked_add(padding_out))
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d output dimensions overflow"))?;
        let trim = padding
            .checked_mul(2)
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d padding overflows"))?;
        if untrimmed_length <= trim {
            bail!("Vulkan ConvTranspose1d padding removes all output positions");
        }
        let output_length = untrimmed_length - trim;
        let output_count = output_channels
            .checked_mul(output_length)
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d output dimensions overflow"))?;
        let zero_bias = vec![0.0_f32; output_channels];
        let bias_values = bias.unwrap_or(&zero_bias);
        let push_constants = [
            u32::try_from(input_channels)
                .wrap_err("ConvTranspose1d input channels are too large")?,
            u32::try_from(input_length).wrap_err("ConvTranspose1d input length is too large")?,
            u32::try_from(output_channels)
                .wrap_err("ConvTranspose1d output channels are too large")?,
            u32::try_from(output_length).wrap_err("ConvTranspose1d output length is too large")?,
            u32::try_from(kernel_size).wrap_err("ConvTranspose1d kernel size is too large")?,
            u32::try_from(stride).wrap_err("ConvTranspose1d stride is too large")?,
            u32::try_from(dilation).wrap_err("ConvTranspose1d dilation is too large")?,
            u32::try_from(padding).wrap_err("ConvTranspose1d padding is too large")?,
            u32::try_from(padding_out).wrap_err("ConvTranspose1d output padding is too large")?,
            u32::from(bias.is_some()),
        ];
        dispatch_four_buffer_kernel(
            &self.device,
            &self.memory_properties,
            self.queue,
            self.queue_family_index,
            self.conv_transpose1d_pipeline.descriptor_layout,
            self.conv_transpose1d_pipeline.pipeline_layout,
            self.conv_transpose1d_pipeline.pipeline,
            weights,
            bias_values,
            input,
            output_count,
            &push_constants,
            "ConvTranspose1d",
        )
    }

    /// Run a batch-one, ungrouped float32 Conv1d through the cached Vulkan
    /// pipeline. The host-slice boundary is intentional for this first model
    /// kernel; persistent tensor-resident chaining is the next optimization.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes, padding, buffer creation, dispatch, or
    /// result transfer is invalid.
    #[expect(
        clippy::too_many_arguments,
        reason = "The kernel boundary exposes the fixed-shape convolution parameters explicitly."
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "The first Conv1d dispatch keeps Vulkan resource lifetimes auditable."
    )]
    #[expect(
        clippy::semicolon_if_nothing_returned,
        reason = "Ash command recording remains explicit inside documented unsafe blocks."
    )]
    pub fn dispatch_conv1d(
        &self,
        weights: &[f32],
        output_channels: usize,
        input_channels: usize,
        kernel_size: usize,
        bias: Option<&[f32]>,
        input: &[f32],
        input_length: usize,
        stride: usize,
        dilation: usize,
        padding_left: usize,
        padding_right: usize,
    ) -> Result<Vec<f32>> {
        if output_channels == 0 || input_channels == 0 || kernel_size == 0 {
            bail!("Vulkan Conv1d dimensions must be non-zero");
        }
        if stride == 0 || dilation == 0 {
            bail!("Vulkan Conv1d stride and dilation must be non-zero");
        }
        let expected_weights = output_channels
            .checked_mul(input_channels)
            .and_then(|value| value.checked_mul(kernel_size))
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d weight dimensions overflow"))?;
        if weights.len() != expected_weights {
            bail!(
                "Vulkan Conv1d weights have {}, expected {}",
                weights.len(),
                expected_weights
            );
        }
        let expected_input = input_channels
            .checked_mul(input_length)
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d input dimensions overflow"))?;
        if input.len() != expected_input {
            bail!(
                "Vulkan Conv1d input has {}, expected {}",
                input.len(),
                expected_input
            );
        }
        if let Some(bias) = bias
            && bias.len() != output_channels
        {
            bail!(
                "Vulkan Conv1d bias has {}, expected {}",
                bias.len(),
                output_channels
            );
        }
        let effective_kernel = dilation
            .checked_mul(kernel_size.saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d effective kernel overflows"))?;
        let padded_input = input_length
            .checked_add(padding_left)
            .and_then(|value| value.checked_add(padding_right))
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d padded input overflows"))?;
        if padded_input < effective_kernel {
            bail!("Vulkan Conv1d padding produces no output positions");
        }
        let output_length = (padded_input - effective_kernel) / stride + 1;
        let output_count = output_channels
            .checked_mul(output_length)
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d output dimensions overflow"))?;
        let weight_bytes = u64::try_from(std::mem::size_of_val(weights))
            .wrap_err("Vulkan Conv1d weights are too large")?;
        let bias_values = bias.unwrap_or(&[]);
        let has_bias = bias.is_some();
        let bias_bytes = u64::try_from(output_channels * std::mem::size_of::<f32>())
            .wrap_err("Vulkan Conv1d bias is too large")?;
        let input_bytes = u64::try_from(std::mem::size_of_val(input))
            .wrap_err("Vulkan Conv1d input is too large")?;
        let output_bytes = u64::try_from(output_count * std::mem::size_of::<f32>())
            .wrap_err("Vulkan Conv1d output is too large")?;
        let weight_buffer = create_buffer(
            &self.device,
            &self.memory_properties,
            weight_bytes,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let bias_buffer = create_buffer(
            &self.device,
            &self.memory_properties,
            bias_bytes,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let input_buffer = create_buffer(
            &self.device,
            &self.memory_properties,
            input_bytes,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let output_buffer = create_buffer(
            &self.device,
            &self.memory_properties,
            output_bytes,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        write_buffer_bytes(
            &self.device,
            &weight_buffer,
            weights_as_bytes(weights),
            weight_bytes,
        )?;
        if has_bias {
            write_buffer_bytes(
                &self.device,
                &bias_buffer,
                weights_as_bytes(bias_values),
                bias_bytes,
            )?;
        } else {
            write_buffer_bytes(
                &self.device,
                &bias_buffer,
                weights_as_bytes(&vec![0.0_f32; output_channels]),
                bias_bytes,
            )?;
        }
        write_buffer_bytes(
            &self.device,
            &input_buffer,
            weights_as_bytes(input),
            input_bytes,
        )?;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4);
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        // SAFETY: Descriptor pool data remains alive for the call.
        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .wrap_err("failed to create the Vulkan Conv1d descriptor pool")?;
        let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(
                &self.conv1d_pipeline.descriptor_layout,
            ));
        // SAFETY: The pool and layout belong to this live device.
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&descriptor_set_info) }
            .wrap_err("failed to allocate the Vulkan Conv1d descriptor set")?
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("Vulkan returned no Conv1d descriptor set"))?;
        let weight_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(weight_buffer.buffer)
            .range(weight_bytes);
        let bias_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(bias_buffer.buffer)
            .range(bias_bytes);
        let input_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(input_buffer.buffer)
            .range(input_bytes);
        let output_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(output_buffer.buffer)
            .range(output_bytes);
        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&weight_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&bias_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&input_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&output_descriptor)),
        ];
        // SAFETY: Descriptor handles and buffer infos belong to this live device.
        unsafe { self.device.update_descriptor_sets(&descriptor_writes, &[]) };
        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family_index);
        // SAFETY: The queue family was selected from this physical device.
        let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }
            .wrap_err("failed to create the Vulkan Conv1d command pool")?;
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The command pool belongs to this live device.
        let command_buffer = unsafe { self.device.allocate_command_buffers(&command_buffer_info) }
            .wrap_err("failed to allocate the Vulkan Conv1d command buffer")?
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("Vulkan returned no Conv1d command buffer"))?;
        // SAFETY: The command buffer was allocated from this live command pool.
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
        }
        .wrap_err("failed to begin the Vulkan Conv1d command buffer")?;
        // SAFETY: The pipeline and descriptor set belong to this live device.
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.conv1d_pipeline.pipeline,
            )
        };
        // SAFETY: The descriptor set layout matches the Conv1d shader.
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.conv1d_pipeline.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            )
        };
        let push_constants = [
            u32::try_from(input_channels).wrap_err("Conv1d input channels are too large")?,
            u32::try_from(input_length).wrap_err("Conv1d input length is too large")?,
            u32::try_from(output_channels).wrap_err("Conv1d output channels are too large")?,
            u32::try_from(output_length).wrap_err("Conv1d output length is too large")?,
            u32::try_from(kernel_size).wrap_err("Conv1d kernel size is too large")?,
            u32::try_from(stride).wrap_err("Conv1d stride is too large")?,
            u32::try_from(dilation).wrap_err("Conv1d dilation is too large")?,
            u32::try_from(padding_left).wrap_err("Conv1d left padding is too large")?,
            u32::from(has_bias),
            0,
        ];
        // SAFETY: The push-constant byte slice views a live array matching the
        // pipeline layout range.
        let push_constant_bytes = unsafe {
            std::slice::from_raw_parts(
                push_constants.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&push_constants),
            )
        };
        // SAFETY: The push constants match the Conv1d pipeline layout.
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.conv1d_pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constant_bytes,
            )
        };
        let workgroups = u32::try_from(output_count.div_ceil(256))
            .wrap_err("Vulkan Conv1d dispatch dimensions are too large")?;
        // SAFETY: The shader bounds-checks all output invocations.
        unsafe { self.device.cmd_dispatch(command_buffer, workgroups, 1, 1) };
        // SAFETY: The command buffer recording has a valid begin/dispatch sequence.
        unsafe { self.device.end_command_buffer(command_buffer) }
            .wrap_err("failed to end the Vulkan Conv1d command buffer")?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        // SAFETY: The fence belongs to this live device.
        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .wrap_err("failed to create the Vulkan Conv1d fence")?;
        // SAFETY: The submit info references the recorded command buffer.
        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
        }
        .wrap_err("failed to submit the Vulkan Conv1d command buffer")?;
        // SAFETY: The fence belongs to this live device and was signaled by
        // the submitted command buffer.
        unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) }
            .wrap_err("failed while waiting for the Vulkan Conv1d fence")?;
        let output = read_buffer(&self.device, &output_buffer, output_bytes, output_count)?;
        // SAFETY: All handles below are no longer referenced after the fence wait.
        unsafe { self.device.destroy_fence(fence, None) };
        // SAFETY: The command pool is no longer in use after the fence wait.
        unsafe { self.device.destroy_command_pool(command_pool, None) };
        // SAFETY: Descriptor objects are no longer used by submitted work.
        unsafe { self.device.destroy_descriptor_pool(descriptor_pool, None) };
        destroy_buffer(&self.device, &weight_buffer);
        destroy_buffer(&self.device, &bias_buffer);
        destroy_buffer(&self.device, &input_buffer);
        destroy_buffer(&self.device, &output_buffer);
        Ok(output)
    }

    /// Run a model embedding lookup through a persistent Vulkan device.
    ///
    /// The method is the first model-shaped kernel in the Vulkan candidate.
    /// It intentionally accepts ordinary host slices so the artifact and
    /// parity boundary can be tested before the full graph is moved on-device.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions, token IDs, shader creation, dispatch,
    /// or result transfer is invalid.
    ///
    /// # Panics
    ///
    /// Panics only if a checked byte-count conversion or fixed shader layout
    /// conversion cannot fit the Vulkan integer widths.
    #[expect(
        clippy::too_many_lines,
        reason = "The first persistent model kernel keeps Vulkan lifetime and cleanup auditable."
    )]
    #[expect(
        clippy::semicolon_if_nothing_returned,
        reason = "Ash command recording remains explicit inside documented unsafe blocks."
    )]
    pub fn dispatch_embedding(
        &self,
        weights: &[f32],
        vocabulary_size: usize,
        embedding_dimension: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            bail!("Vulkan embedding requires at least one token");
        }
        if tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |token| token >= vocabulary_size))
        {
            bail!("Vulkan embedding token ID exceeds the prepared vocabulary");
        }
        let expected_weights = vocabulary_size
            .checked_mul(embedding_dimension)
            .ok_or_else(|| eyre::eyre!("Vulkan embedding dimensions overflow"))?;
        let owned_weight_buffer = match &self.embedding_weights {
            None => {
                if weights.len() != expected_weights {
                    bail!(
                        "Vulkan embedding weights have {}, expected {}",
                        weights.len(),
                        expected_weights
                    );
                }
                let weight_bytes = u64::try_from(std::mem::size_of_val(weights))
                    .wrap_err("Vulkan embedding weights are too large")?;
                let weight_buffer = create_buffer(
                    &self.device,
                    &self.memory_properties,
                    weight_bytes,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                write_buffer_bytes(
                    &self.device,
                    &weight_buffer,
                    weights_as_bytes(weights),
                    weight_bytes,
                )?;
                Some(weight_buffer)
            }
            Some(prepared) => {
                if prepared.vocabulary_size != vocabulary_size
                    || prepared.embedding_dimension != embedding_dimension
                {
                    bail!(
                        "prepared Vulkan embedding shape is {}x{}, requested {}x{}",
                        prepared.vocabulary_size,
                        prepared.embedding_dimension,
                        vocabulary_size,
                        embedding_dimension
                    );
                }
                None
            }
        };
        let (weight_buffer, weight_bytes) = if let Some(prepared) = &self.embedding_weights {
            (&prepared.buffer, prepared.byte_len)
        } else {
            let weight_buffer = owned_weight_buffer
                .as_ref()
                .expect("an uncached embedding owns its temporary weight buffer");
            (
                weight_buffer,
                u64::try_from(std::mem::size_of_val(weights)).expect("fits u64"),
            )
        };
        let output_count = tokens
            .len()
            .checked_mul(embedding_dimension)
            .ok_or_else(|| eyre::eyre!("Vulkan embedding output dimensions overflow"))?;
        let token_bytes = std::mem::size_of_val(tokens);
        let output_bytes = output_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| eyre::eyre!("Vulkan embedding output byte count overflows"))?;
        let token_buffer = create_buffer(
            &self.device,
            &self.memory_properties,
            u64::try_from(token_bytes).wrap_err("Vulkan embedding tokens are too large")?,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let output_buffer = create_buffer(
            &self.device,
            &self.memory_properties,
            u64::try_from(output_bytes).wrap_err("Vulkan embedding output is too large")?,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        write_buffer_bytes(
            &self.device,
            &token_buffer,
            tokens_as_bytes(tokens),
            u64::try_from(token_bytes).expect("token byte count fits u64"),
        )?;

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3);
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        // SAFETY: Descriptor pool data remains alive for the call.
        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .wrap_err("failed to create the Vulkan embedding descriptor pool")?;
        let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(
                &self.embedding_pipeline.descriptor_layout,
            ));
        // SAFETY: The pool and layout belong to this live device.
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&descriptor_set_info) }
            .wrap_err("failed to allocate the Vulkan embedding descriptor set")?
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("Vulkan returned no embedding descriptor set"))?;
        let weight_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(weight_buffer.buffer)
            .range(weight_bytes);
        let token_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(token_buffer.buffer)
            .range(u64::try_from(token_bytes).expect("token byte count fits u64"));
        let output_descriptor = vk::DescriptorBufferInfo::default()
            .buffer(output_buffer.buffer)
            .range(u64::try_from(output_bytes).expect("output byte count fits u64"));
        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&weight_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&token_descriptor)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&output_descriptor)),
        ];
        // SAFETY: Descriptor handles and buffer infos belong to this live device.
        unsafe { self.device.update_descriptor_sets(&descriptor_writes, &[]) };
        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family_index);
        // SAFETY: The queue family was selected from this physical device.
        let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }
            .wrap_err("failed to create the Vulkan embedding command pool")?;
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The command pool belongs to this live device.
        let command_buffer = unsafe { self.device.allocate_command_buffers(&command_buffer_info) }
            .wrap_err("failed to allocate the Vulkan embedding command buffer")?
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("Vulkan returned no embedding command buffer"))?;
        // SAFETY: The command buffer was allocated from this live command pool.
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
        }
        .wrap_err("failed to begin the Vulkan embedding command buffer")?;
        // SAFETY: The pipeline and descriptor set belong to this live device.
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.embedding_pipeline.pipeline,
            )
        };
        // SAFETY: The descriptor set layout matches the embedding shader.
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.embedding_pipeline.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            )
        };
        let push_constants = [
            u32::try_from(tokens.len()).wrap_err("embedding sequence is too long")?,
            u32::try_from(embedding_dimension).wrap_err("embedding dimension is too large")?,
        ];
        // SAFETY: The push-constant byte slice is a view of a live two-u32
        // array and matches the pipeline layout range.
        let push_constant_bytes = unsafe {
            std::slice::from_raw_parts(
                push_constants.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&push_constants),
            )
        };
        // SAFETY: The push constants match the compute pipeline layout.
        unsafe {
            self.device.cmd_push_constants(
                command_buffer,
                self.embedding_pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constant_bytes,
            )
        };
        let workgroups = u32::try_from(output_count.div_ceil(256))
            .wrap_err("Vulkan embedding dispatch dimensions are too large")?;
        // SAFETY: The shader bounds-checks all invocations against output_count.
        unsafe { self.device.cmd_dispatch(command_buffer, workgroups, 1, 1) };
        // SAFETY: The command buffer recording has a valid begin/dispatch sequence.
        unsafe { self.device.end_command_buffer(command_buffer) }
            .wrap_err("failed to end the Vulkan embedding command buffer")?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        // SAFETY: The fence belongs to this live device.
        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .wrap_err("failed to create the Vulkan embedding fence")?;
        // SAFETY: The submit info references the recorded command buffer.
        unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
        }
        .wrap_err("failed to submit the Vulkan embedding command buffer")?;
        // SAFETY: The fence belongs to this live device and was signaled by
        // the submitted command buffer.
        unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) }
            .wrap_err("failed while waiting for the Vulkan embedding fence")?;
        let output = read_buffer(
            &self.device,
            &output_buffer,
            u64::try_from(output_bytes).expect("output byte count fits u64"),
            output_count,
        )?;
        // SAFETY: All handles below are no longer referenced after the fence
        // wait.
        unsafe { self.device.destroy_fence(fence, None) };
        // SAFETY: The command pool is no longer in use after the fence wait.
        unsafe { self.device.destroy_command_pool(command_pool, None) };
        // SAFETY: Descriptor objects are no longer used by submitted work.
        unsafe { self.device.destroy_descriptor_pool(descriptor_pool, None) };
        if let Some(weight_buffer) = owned_weight_buffer {
            destroy_buffer(&self.device, &weight_buffer);
        }
        destroy_buffer(&self.device, &token_buffer);
        destroy_buffer(&self.device, &output_buffer);
        Ok(output)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "The shared four-buffer dispatch helper receives the explicit Ash handles it operates on."
)]
#[expect(
    clippy::too_many_lines,
    reason = "The shared model-kernel dispatch keeps resource lifetimes auditable."
)]
#[expect(
    clippy::semicolon_if_nothing_returned,
    reason = "Ash command recording remains explicit inside documented unsafe blocks."
)]
fn dispatch_four_buffer_kernel(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    queue_family_index: u32,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    output_count: usize,
    push_constants: &[u32],
    operation_name: &str,
) -> Result<Vec<f32>> {
    let weight_bytes = u64::try_from(std::mem::size_of_val(weights))
        .wrap_err_with(|| format!("Vulkan {operation_name} weights are too large"))?;
    let bias_bytes = u64::try_from(std::mem::size_of_val(bias))
        .wrap_err_with(|| format!("Vulkan {operation_name} bias is too large"))?;
    let input_bytes = u64::try_from(std::mem::size_of_val(input))
        .wrap_err_with(|| format!("Vulkan {operation_name} input is too large"))?;
    let output_bytes = u64::try_from(
        output_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| eyre::eyre!("Vulkan {operation_name} output byte count overflows"))?,
    )
    .wrap_err_with(|| format!("Vulkan {operation_name} output is too large"))?;
    let weight_buffer = create_buffer(
        device,
        memory_properties,
        weight_bytes,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let bias_buffer = create_buffer(
        device,
        memory_properties,
        bias_bytes,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let input_buffer = create_buffer(
        device,
        memory_properties,
        input_bytes,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let output_buffer = create_buffer(
        device,
        memory_properties,
        output_bytes,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    write_buffer_bytes(
        device,
        &weight_buffer,
        weights_as_bytes(weights),
        weight_bytes,
    )?;
    write_buffer_bytes(device, &bias_buffer, weights_as_bytes(bias), bias_bytes)?;
    write_buffer_bytes(device, &input_buffer, weights_as_bytes(input), input_bytes)?;
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(4);
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(std::slice::from_ref(&pool_size));
    // SAFETY: Descriptor pool data remains alive for the call.
    let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }
        .wrap_err_with(|| {
            format!("failed to create the Vulkan {operation_name} descriptor pool")
        })?;
    let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_layout));
    // SAFETY: The pool and layout belong to this live device.
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&descriptor_set_info) }
        .wrap_err_with(|| format!("failed to allocate the Vulkan {operation_name} descriptor set"))?
        .into_iter()
        .next()
        .ok_or_else(|| eyre::eyre!("Vulkan returned no {operation_name} descriptor set"))?;
    let weight_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(weight_buffer.buffer)
        .range(weight_bytes);
    let bias_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(bias_buffer.buffer)
        .range(bias_bytes);
    let input_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(input_buffer.buffer)
        .range(input_bytes);
    let output_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(output_buffer.buffer)
        .range(output_bytes);
    let descriptor_writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&weight_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&bias_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&input_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&output_descriptor)),
    ];
    // SAFETY: Descriptor handles and buffer infos belong to this live device.
    unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
    let command_pool_info =
        vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
    // SAFETY: The queue family was selected from this physical device.
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
        .wrap_err_with(|| format!("failed to create the Vulkan {operation_name} command pool"))?;
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: The command pool belongs to this live device.
    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_info) }
        .wrap_err_with(|| format!("failed to allocate the Vulkan {operation_name} command buffer"))?
        .into_iter()
        .next()
        .ok_or_else(|| eyre::eyre!("Vulkan returned no {operation_name} command buffer"))?;
    // SAFETY: The command buffer was allocated from this live command pool.
    unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()) }
        .wrap_err_with(|| format!("failed to begin the Vulkan {operation_name} command buffer"))?;
    // SAFETY: The pipeline and descriptor set belong to this live device.
    unsafe { device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline) };
    // SAFETY: The descriptor set layout matches the selected model kernel.
    unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        )
    };
    // SAFETY: The push-constant byte slice views a live u32 array and matches
    // the pipeline layout range.
    let push_constant_bytes = unsafe {
        std::slice::from_raw_parts(
            push_constants.as_ptr().cast::<u8>(),
            std::mem::size_of_val(push_constants),
        )
    };
    // SAFETY: The push constants match the selected model kernel layout.
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_constant_bytes,
        )
    };
    let workgroups = u32::try_from(output_count.div_ceil(256))
        .wrap_err_with(|| format!("Vulkan {operation_name} dispatch dimensions are too large"))?;
    // SAFETY: The shader bounds-checks all output invocations.
    unsafe { device.cmd_dispatch(command_buffer, workgroups, 1, 1) };
    // SAFETY: The command buffer recording has a valid begin/dispatch sequence.
    unsafe { device.end_command_buffer(command_buffer) }
        .wrap_err_with(|| format!("failed to end the Vulkan {operation_name} command buffer"))?;
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    // SAFETY: The fence belongs to this live device.
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .wrap_err_with(|| format!("failed to create the Vulkan {operation_name} fence"))?;
    // SAFETY: The submit info references the recorded command buffer.
    unsafe { device.queue_submit(queue, std::slice::from_ref(&submit_info), fence) }
        .wrap_err_with(|| format!("failed to submit the Vulkan {operation_name} command buffer"))?;
    // SAFETY: The fence belongs to this live device and was signaled by the
    // submitted command buffer.
    unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
        .wrap_err_with(|| format!("failed while waiting for the Vulkan {operation_name} fence"))?;
    let output = read_buffer(device, &output_buffer, output_bytes, output_count)?;
    // SAFETY: All handles below are no longer referenced after the fence wait.
    unsafe { device.destroy_fence(fence, None) };
    // SAFETY: The command pool is no longer in use after the fence wait.
    unsafe { device.destroy_command_pool(command_pool, None) };
    // SAFETY: Descriptor objects are no longer used by submitted work.
    unsafe { device.destroy_descriptor_pool(descriptor_pool, None) };
    destroy_buffer(device, &weight_buffer);
    destroy_buffer(device, &bias_buffer);
    destroy_buffer(device, &input_buffer);
    destroy_buffer(device, &output_buffer);
    Ok(output)
}

const ELEMENTWISE_LEAKY_RELU: u32 = 0;
const ELEMENTWISE_ADD: u32 = 1;
const ELEMENTWISE_SCALE: u32 = 2;
const ELEMENTWISE_TANH: u32 = 3;
const ELEMENTWISE_MUL: u32 = 4;
const ELEMENTWISE_ONE_MINUS: u32 = 5;
const ELEMENTWISE_RELU: u32 = 6;
const ELEMENTWISE_SIGMOID: u32 = 7;
const BATCH_BUFFER_ALIGNMENT: vk::DeviceSize = 256;
const INITIAL_BATCH_ARENA_BYTES: vk::DeviceSize = 16 * 1024 * 1024;
const DESCRIPTOR_SET_ALLOCATION_BLOCK: usize = 32;

fn profile_batches_enabled() -> bool {
    matches!(
        std::env::var("TEAMY_TTS_VULKAN_PROFILE").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

#[derive(Clone, Copy)]
struct StorageBufferBinding {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
}

struct BatchArena {
    allocation: BufferAllocation,
    capacity: vk::DeviceSize,
    used: vk::DeviceSize,
}

#[derive(Clone, Copy)]
struct BatchBufferSlice {
    arena_index: usize,
    offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

struct DescriptorSetCache {
    layout: vk::DescriptorSetLayout,
    free: Vec<vk::DescriptorSet>,
}

/// One recorded model-inference submission with device-resident intermediates.
///
/// The batch intentionally owns temporary buffers and descriptor state for a
/// single utterance. Its command buffer is submitted once, after all model
/// operations have been recorded, so host synchronization does not occur
/// between adjacent convolutions.
pub(crate) struct VulkanBatch<'a> {
    context: &'a VulkanContext,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    fence: vk::Fence,
    timestamp_query_pool: Option<vk::QueryPool>,
    record_started: std::time::Instant,
    arenas: Vec<BatchArena>,
    buffers: Vec<BatchBufferSlice>,
    descriptor_sets: Vec<DescriptorSetCache>,
    staging_buffers: Vec<BufferAllocation>,
    readback_buffers: Vec<BufferAllocation>,
}

impl<'a> VulkanBatch<'a> {
    fn new(context: &'a VulkanContext) -> Result<Self> {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(context.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: The queue family was selected from this live physical device.
        let command_pool = unsafe { context.device.create_command_pool(&command_pool_info, None) }
            .wrap_err("failed to create the Vulkan model batch command pool")?;
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The command pool belongs to this live device.
        let command_buffer = unsafe {
            context
                .device
                .allocate_command_buffers(&command_buffer_info)
        }
        .wrap_err("failed to allocate the Vulkan model batch command buffer")?
        .into_iter()
        .next()
        .ok_or_else(|| eyre::eyre!("Vulkan returned no model batch command buffer"))?;
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4_096);
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1_024)
            .pool_sizes(std::slice::from_ref(&pool_size));
        // SAFETY: Descriptor pool data remains alive for the call.
        let descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .wrap_err("failed to create the Vulkan model batch descriptor pool")?;
        // SAFETY: The command buffer belongs to the newly created pool.
        unsafe {
            context
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
        }
        .wrap_err("failed to begin the Vulkan model batch command buffer")?;
        let timestamp_query_pool = if context.profile_batches && context.timestamp_valid_bits > 0 {
            Some(create_timestamp_query_pool(&context.device)?)
        } else {
            None
        };
        if let Some(query_pool) = timestamp_query_pool {
            // SAFETY: The query pool belongs to this live device and this
            // command buffer is in the recording state.
            let () = unsafe {
                context
                    .device
                    .cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
            };
            // SAFETY: The query pool belongs to this live device and the
            // timestamp is recorded in this command buffer.
            unsafe {
                context.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    0,
                );
            }
        }
        Ok(Self {
            context,
            command_pool,
            command_buffer,
            descriptor_pool,
            fence: vk::Fence::null(),
            timestamp_query_pool,
            record_started: std::time::Instant::now(),
            arenas: Vec::new(),
            buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            staging_buffers: Vec::new(),
            readback_buffers: Vec::new(),
        })
    }

    pub(crate) fn alloc_buffer(&mut self, element_count: usize) -> Result<VulkanBatchBufferHandle> {
        if element_count == 0 {
            bail!("Vulkan batch buffers cannot be empty");
        }
        let byte_len = u64::try_from(
            element_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| eyre::eyre!("Vulkan batch buffer size overflows usize"))?,
        )
        .wrap_err("Vulkan batch buffer is too large")?;
        let aligned_byte_len = byte_len
            .checked_add(BATCH_BUFFER_ALIGNMENT - 1)
            .ok_or_else(|| eyre::eyre!("Vulkan batch buffer alignment overflows"))?
            & !(BATCH_BUFFER_ALIGNMENT - 1);
        let (arena_index, offset) = match self.arenas.last_mut() {
            Some(arena) if arena.capacity - arena.used >= aligned_byte_len => {
                let offset = arena.used;
                arena.used += aligned_byte_len;
                (self.arenas.len() - 1, offset)
            }
            _ => {
                let capacity = INITIAL_BATCH_ARENA_BYTES.max(aligned_byte_len);
                let allocation = create_buffer_with_usage(
                    &self.context.device,
                    &self.context.memory_properties,
                    capacity,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                )?;
                self.arenas.push(BatchArena {
                    allocation,
                    capacity,
                    used: aligned_byte_len,
                });
                (self.arenas.len() - 1, 0)
            }
        };
        let index = self.buffers.len();
        self.buffers.push(BatchBufferSlice {
            arena_index,
            offset,
            byte_len,
        });
        Ok(VulkanBatchBufferHandle(index))
    }

    pub(crate) fn write_buffer(
        &mut self,
        handle: VulkanBatchBufferHandle,
        values: &[f32],
    ) -> Result<()> {
        let (target_buffer, target_offset, capacity) = {
            let (allocation, offset, capacity) = self.batch_allocation(handle)?;
            (allocation.buffer, offset, capacity)
        };
        let byte_len = u64::try_from(std::mem::size_of_val(values))
            .wrap_err("Vulkan batch input is too large")?;
        if byte_len > capacity {
            bail!("Vulkan batch input exceeds its allocated buffer");
        }
        let staging = create_buffer_with_usage(
            &self.context.device,
            &self.context.memory_properties,
            byte_len,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        write_buffer_bytes(
            &self.context.device,
            &staging,
            weights_as_bytes(values),
            byte_len,
        )?;
        let copy = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(target_offset)
            .size(byte_len);
        // SAFETY: The staging and arena buffers belong to this live device,
        // and the copy range is within both allocations.
        unsafe {
            self.context.device.cmd_copy_buffer(
                self.command_buffer,
                staging.buffer,
                target_buffer,
                std::slice::from_ref(&copy),
            );
        };
        self.add_transfer_to_compute_barrier();
        self.staging_buffers.push(staging);
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded linear boundary exposes fixed-shape dimensions and persistent handles explicitly."
    )]
    pub(crate) fn dispatch_linear(
        &mut self,
        weights: VulkanModelBufferHandle,
        bias: VulkanModelBufferHandle,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_channels: usize,
        sequence_length: usize,
        output_channels: usize,
        has_bias: bool,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(input_channels).wrap_err("linear input channels are too large")?,
            u32::try_from(sequence_length).wrap_err("linear sequence length is too large")?,
            u32::try_from(output_channels).wrap_err("linear output channels are too large")?,
            u32::from(has_bias),
        ];
        let weights = self.model_binding(weights)?;
        let bias = self.model_binding(bias)?;
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.linear_pipeline.descriptor_layout,
            &[weights, bias, input, output],
        )?;
        self.record_four_buffer_dispatch(
            self.context.linear_pipeline.pipeline,
            self.context.linear_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            output_channels
                .checked_mul(sequence_length)
                .ok_or_else(|| eyre::eyre!("linear dispatch dimensions overflow"))?,
        )
    }

    pub(crate) fn dispatch_length_regulate(
        &mut self,
        input: VulkanBatchBufferHandle,
        durations: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        channels: usize,
        token_count: usize,
        frame_count: usize,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(channels).wrap_err("length-regulation channels are too large")?,
            u32::try_from(token_count).wrap_err("length-regulation token count is too large")?,
            u32::try_from(frame_count).wrap_err("length-regulation frame count is too large")?,
        ];
        let input_handle = input;
        let duration_handle = durations;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let durations = self.batch_binding(duration_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.length_regulate_pipeline.descriptor_layout,
            &[input, durations, output],
        )?;
        self.record_dispatch(
            self.context.length_regulate_pipeline.pipeline,
            self.context.length_regulate_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            channels
                .checked_mul(frame_count)
                .ok_or_else(|| eyre::eyre!("length-regulation dispatch dimensions overflow"))?,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded LSTM boundary exposes fixed model dimensions and persistent handles explicitly."
    )]
    pub(crate) fn dispatch_lstm(
        &mut self,
        input_weights: VulkanModelBufferHandle,
        hidden_weights: VulkanModelBufferHandle,
        bias: VulkanModelBufferHandle,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_channels: usize,
        hidden_channels: usize,
        sequence_length: usize,
        reverse: bool,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(input_channels).wrap_err("LSTM input channels are too large")?,
            u32::try_from(hidden_channels).wrap_err("LSTM hidden channels are too large")?,
            u32::try_from(sequence_length).wrap_err("LSTM sequence length is too large")?,
            u32::from(reverse),
        ];
        let input_weights = self.model_binding(input_weights)?;
        let hidden_weights = self.model_binding(hidden_weights)?;
        let bias = self.model_binding(bias)?;
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.lstm_pipeline.descriptor_layout,
            &[input_weights, hidden_weights, bias, input, output],
        )?;
        self.record_dispatch_with_workgroups(
            self.context.lstm_pipeline.pipeline,
            self.context.lstm_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            1,
        );
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded batch-normalization boundary exposes persistent statistics and fixed-shape dimensions explicitly."
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The Vulkan batch-normalization shader consumes f32 epsilon, matching Burn's inference arithmetic."
    )]
    pub(crate) fn dispatch_batch_norm(
        &mut self,
        gamma: VulkanModelBufferHandle,
        beta: VulkanModelBufferHandle,
        running_mean: VulkanModelBufferHandle,
        running_variance: VulkanModelBufferHandle,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        channels: usize,
        sequence_length: usize,
        epsilon: f64,
    ) -> Result<()> {
        let epsilon = epsilon as f32;
        if !epsilon.is_finite() {
            bail!("Vulkan batch-normalization epsilon must be finite");
        }
        let push_constants = [
            u32::try_from(channels).wrap_err("batch-normalization channels are too large")?,
            u32::try_from(sequence_length)
                .wrap_err("batch-normalization sequence length is too large")?,
            epsilon.to_bits(),
        ];
        let gamma = self.model_binding(gamma)?;
        let beta = self.model_binding(beta)?;
        let running_mean = self.model_binding(running_mean)?;
        let running_variance = self.model_binding(running_variance)?;
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.batch_norm_pipeline.descriptor_layout,
            &[gamma, beta, running_mean, running_variance, input, output],
        )?;
        self.record_dispatch(
            self.context.batch_norm_pipeline.pipeline,
            self.context.batch_norm_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            channels
                .checked_mul(sequence_length)
                .ok_or_else(|| eyre::eyre!("batch-normalization dispatch dimensions overflow"))?,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded MaxPool1d boundary exposes fixed model dimensions explicitly."
    )]
    pub(crate) fn dispatch_max_pool1d(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        channels: usize,
        input_length: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        padding_left: usize,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(channels).wrap_err("MaxPool1d channels are too large")?,
            u32::try_from(input_length).wrap_err("MaxPool1d input length is too large")?,
            u32::try_from(output_length).wrap_err("MaxPool1d output length is too large")?,
            u32::try_from(kernel_size).wrap_err("MaxPool1d kernel size is too large")?,
            u32::try_from(stride).wrap_err("MaxPool1d stride is too large")?,
            u32::try_from(padding_left).wrap_err("MaxPool1d padding is too large")?,
        ];
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.max_pool1d_pipeline.descriptor_layout,
            &[input, output],
        )?;
        self.record_dispatch(
            self.context.max_pool1d_pipeline.pipeline,
            self.context.max_pool1d_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            channels
                .checked_mul(output_length)
                .ok_or_else(|| eyre::eyre!("MaxPool1d dispatch dimensions overflow"))?,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded GRU boundary exposes separate reset-after biases and fixed-shape dimensions explicitly."
    )]
    pub(crate) fn dispatch_gru(
        &mut self,
        input_weights: VulkanModelBufferHandle,
        hidden_weights: VulkanModelBufferHandle,
        input_bias: VulkanModelBufferHandle,
        hidden_bias: VulkanModelBufferHandle,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_channels: usize,
        hidden_channels: usize,
        sequence_length: usize,
        reverse: bool,
        reset_after: bool,
    ) -> Result<()> {
        if hidden_channels > 512 {
            bail!(
                "Vulkan GRU hidden size {} exceeds the fixed kernel limit 512",
                hidden_channels
            );
        }
        let push_constants = [
            u32::try_from(input_channels).wrap_err("GRU input channels are too large")?,
            u32::try_from(hidden_channels).wrap_err("GRU hidden channels are too large")?,
            u32::try_from(sequence_length).wrap_err("GRU sequence length is too large")?,
            u32::from(reverse),
            u32::from(reset_after),
        ];
        let input_weights = self.model_binding(input_weights)?;
        let hidden_weights = self.model_binding(hidden_weights)?;
        let input_bias = self.model_binding(input_bias)?;
        let hidden_bias = self.model_binding(hidden_bias)?;
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.gru_pipeline.descriptor_layout,
            &[
                input_weights,
                hidden_weights,
                input_bias,
                hidden_bias,
                input,
                output,
            ],
        )?;
        self.record_dispatch_with_workgroups(
            self.context.gru_pipeline.pipeline,
            self.context.gru_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            1,
        );
        Ok(())
    }

    pub(crate) fn dispatch_copy_channels(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        source_channels: usize,
        sequence_length: usize,
        destination_channel_offset: usize,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(source_channels)
                .wrap_err("channel-copy source channels are too large")?,
            u32::try_from(sequence_length).wrap_err("channel-copy sequence length is too large")?,
            u32::try_from(destination_channel_offset)
                .wrap_err("channel-copy destination offset is too large")?,
        ];
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.copy_channels_pipeline.descriptor_layout,
            &[input, output],
        )?;
        self.record_dispatch(
            self.context.copy_channels_pipeline.pipeline,
            self.context.copy_channels_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            source_channels
                .checked_mul(sequence_length)
                .ok_or_else(|| eyre::eyre!("channel-copy dispatch dimensions overflow"))?,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded Conv1d boundary exposes fixed model dimensions explicitly."
    )]
    pub(crate) fn dispatch_conv1d(
        &mut self,
        weight: VulkanModelBufferHandle,
        bias: VulkanModelBufferHandle,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_channels: usize,
        input_length: usize,
        output_channels: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding_left: usize,
        has_bias: bool,
        relu: bool,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(input_channels).wrap_err("Conv1d input channels are too large")?,
            u32::try_from(input_length).wrap_err("Conv1d input length is too large")?,
            u32::try_from(output_channels).wrap_err("Conv1d output channels are too large")?,
            u32::try_from(output_length).wrap_err("Conv1d output length is too large")?,
            u32::try_from(kernel_size).wrap_err("Conv1d kernel size is too large")?,
            u32::try_from(stride).wrap_err("Conv1d stride is too large")?,
            u32::try_from(dilation).wrap_err("Conv1d dilation is too large")?,
            u32::try_from(padding_left).wrap_err("Conv1d padding is too large")?,
            u32::from(has_bias),
            u32::from(relu),
        ];
        let weight = self.model_binding(weight)?;
        let bias = self.model_binding(bias)?;
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.conv1d_pipeline.descriptor_layout,
            &[weight, bias, input, output],
        )?;
        self.record_four_buffer_dispatch(
            self.context.conv1d_pipeline.pipeline,
            self.context.conv1d_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            output_channels
                .checked_mul(output_length)
                .ok_or_else(|| eyre::eyre!("Conv1d dispatch dimensions overflow"))?,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The recorded ConvTranspose1d boundary exposes fixed model dimensions explicitly."
    )]
    pub(crate) fn dispatch_conv_transpose1d(
        &mut self,
        weight: VulkanModelBufferHandle,
        bias: VulkanModelBufferHandle,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_channels: usize,
        input_length: usize,
        output_channels: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        padding_out: usize,
        has_bias: bool,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(input_channels)
                .wrap_err("ConvTranspose1d input channels are too large")?,
            u32::try_from(input_length).wrap_err("ConvTranspose1d input length is too large")?,
            u32::try_from(output_channels)
                .wrap_err("ConvTranspose1d output channels are too large")?,
            u32::try_from(output_length).wrap_err("ConvTranspose1d output length is too large")?,
            u32::try_from(kernel_size).wrap_err("ConvTranspose1d kernel size is too large")?,
            u32::try_from(stride).wrap_err("ConvTranspose1d stride is too large")?,
            u32::try_from(dilation).wrap_err("ConvTranspose1d dilation is too large")?,
            u32::try_from(padding).wrap_err("ConvTranspose1d padding is too large")?,
            u32::try_from(padding_out).wrap_err("ConvTranspose1d output padding is too large")?,
            u32::from(has_bias),
        ];
        let weight = self.model_binding(weight)?;
        let bias = self.model_binding(bias)?;
        let input_handle = input;
        let output_handle = output;
        let input = self.batch_binding(input_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.conv_transpose1d_pipeline.descriptor_layout,
            &[weight, bias, input, output],
        )?;
        self.record_four_buffer_dispatch(
            self.context.conv_transpose1d_pipeline.pipeline,
            self.context.conv_transpose1d_pipeline.pipeline_layout,
            descriptor_set,
            &push_constants,
            output_channels
                .checked_mul(output_length)
                .ok_or_else(|| eyre::eyre!("ConvTranspose1d dispatch dimensions overflow"))?,
        )
    }

    pub(crate) fn dispatch_leaky_relu(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
        negative_slope: f32,
    ) -> Result<()> {
        self.dispatch_elementwise(
            ELEMENTWISE_LEAKY_RELU,
            input,
            input,
            output,
            element_count,
            negative_slope,
        )
    }

    pub(crate) fn dispatch_add(
        &mut self,
        left: VulkanBatchBufferHandle,
        right: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_elementwise(ELEMENTWISE_ADD, left, right, output, element_count, 0.0)
    }

    pub(crate) fn dispatch_scale(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
        scalar: f32,
    ) -> Result<()> {
        self.dispatch_elementwise(
            ELEMENTWISE_SCALE,
            input,
            input,
            output,
            element_count,
            scalar,
        )
    }

    pub(crate) fn dispatch_tanh(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_elementwise(ELEMENTWISE_TANH, input, input, output, element_count, 0.0)
    }

    pub(crate) fn dispatch_mul(
        &mut self,
        left: VulkanBatchBufferHandle,
        right: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_elementwise(ELEMENTWISE_MUL, left, right, output, element_count, 0.0)
    }

    pub(crate) fn dispatch_one_minus(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_elementwise(
            ELEMENTWISE_ONE_MINUS,
            input,
            input,
            output,
            element_count,
            0.0,
        )
    }

    pub(crate) fn dispatch_relu(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_elementwise(ELEMENTWISE_RELU, input, input, output, element_count, 0.0)
    }

    pub(crate) fn dispatch_sigmoid(
        &mut self,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<()> {
        self.dispatch_elementwise(
            ELEMENTWISE_SIGMOID,
            input,
            input,
            output,
            element_count,
            0.0,
        )
    }

    fn dispatch_elementwise(
        &mut self,
        operation: u32,
        input_a: VulkanBatchBufferHandle,
        input_b: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        element_count: usize,
        scalar: f32,
    ) -> Result<()> {
        let push_constants = [
            u32::try_from(element_count).wrap_err("elementwise dispatch is too large")?,
            operation,
            scalar.to_bits(),
        ];
        let left_handle = input_a;
        let right_handle = input_b;
        let output_handle = output;
        let input_a = self.batch_binding(left_handle)?;
        let input_b = self.batch_binding(right_handle)?;
        let output = self.batch_binding(output_handle)?;
        let descriptor_set = self.allocate_descriptor_set(
            self.context.elementwise_pipeline.descriptor_layout,
            &[input_a, input_b, output],
        )?;
        self.record_three_buffer_dispatch(descriptor_set, &push_constants, element_count)
    }

    pub(crate) fn finish(
        self,
        output: VulkanBatchBufferHandle,
        element_count: usize,
    ) -> Result<Vec<f32>> {
        self.finish_outputs(&[(output, element_count)])?
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("Vulkan model batch returned no output"))
    }

    pub(crate) fn finish_outputs(
        mut self,
        outputs: &[(VulkanBatchBufferHandle, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        if let Some(query_pool) = self.timestamp_query_pool {
            // SAFETY: The query pool belongs to this live device and the
            // timestamp is ordered after every recorded dispatch.
            unsafe {
                self.context.device.cmd_write_timestamp(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    1,
                );
            }
        }
        self.add_compute_to_transfer_barrier();
        self.record_output_readbacks(outputs)?;
        self.submit_and_wait()?;
        if let Some(query_pool) = self.timestamp_query_pool {
            let gpu_elapsed_ns = read_timestamp_elapsed(
                &self.context.device,
                query_pool,
                self.context.timestamp_period_ns,
            )?;
            let arena_bytes = self.arenas.iter().map(|arena| arena.capacity).sum::<u64>();
            tracing::info!(
                host_record_elapsed_ns = self.record_started.elapsed().as_nanos(),
                gpu_elapsed_ns,
                temporary_buffer_count = self.buffers.len(),
                arena_count = self.arenas.len(),
                arena_bytes,
                "Vulkan batch profile"
            );
        }
        self.read_outputs(outputs)
    }

    fn record_output_readbacks(
        &mut self,
        outputs: &[(VulkanBatchBufferHandle, usize)],
    ) -> Result<()> {
        for &(output, element_count) in outputs {
            let (source_buffer, source_offset, output_bytes) = {
                let (allocation, offset, byte_len) = self.batch_allocation(output)?;
                let output_bytes = u64::try_from(
                    element_count
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| eyre::eyre!("Vulkan batch output size overflows"))?,
                )
                .wrap_err("Vulkan batch output is too large")?;
                if output_bytes > byte_len {
                    bail!("Vulkan batch output exceeds its allocated buffer");
                }
                (allocation.buffer, offset, output_bytes)
            };
            let staging = create_buffer_with_usage(
                &self.context.device,
                &self.context.memory_properties,
                output_bytes,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::BufferUsageFlags::TRANSFER_DST,
            )?;
            let copy = vk::BufferCopy::default()
                .src_offset(source_offset)
                .dst_offset(0)
                .size(output_bytes);
            // SAFETY: The source and staging buffers belong to this live
            // device, and the copy range is within both allocations.
            unsafe {
                self.context.device.cmd_copy_buffer(
                    self.command_buffer,
                    source_buffer,
                    staging.buffer,
                    std::slice::from_ref(&copy),
                );
            };
            self.readback_buffers.push(staging);
        }
        Ok(())
    }

    fn submit_and_wait(&mut self) -> Result<()> {
        // SAFETY: The command buffer recording contains only valid model
        // dispatches and barriers.
        unsafe { self.context.device.end_command_buffer(self.command_buffer) }
            .wrap_err("failed to end the Vulkan model batch command buffer")?;
        // SAFETY: The fence belongs to this live device.
        self.fence = unsafe {
            self.context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .wrap_err("failed to create the Vulkan model batch fence")?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        // SAFETY: The submit info references the recorded command buffer and
        // the fence belongs to the same live device.
        unsafe {
            self.context.device.queue_submit(
                self.context.queue,
                std::slice::from_ref(&submit_info),
                self.fence,
            )
        }
        .wrap_err("failed to submit the Vulkan model batch")?;
        // SAFETY: The submitted batch has signaled this fence before the
        // output is mapped.
        unsafe {
            self.context
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
        }
        .wrap_err("failed while waiting for the Vulkan model batch")?;
        Ok(())
    }

    fn read_outputs(&self, outputs: &[(VulkanBatchBufferHandle, usize)]) -> Result<Vec<Vec<f32>>> {
        outputs
            .iter()
            .enumerate()
            .map(|(index, (_, element_count))| {
                let staging = self
                    .readback_buffers
                    .get(index)
                    .ok_or_else(|| eyre::eyre!("Vulkan batch readback is missing"))?;
                let output_bytes = u64::try_from(
                    element_count
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| eyre::eyre!("Vulkan batch output size overflows"))?,
                )
                .wrap_err("Vulkan batch output is too large")?;
                read_buffer(&self.context.device, staging, output_bytes, *element_count)
            })
            .collect()
    }

    fn model_buffer(&self, handle: VulkanModelBufferHandle) -> Result<&PersistentModelBuffer> {
        self.context
            .model_buffers
            .get(handle.0)
            .ok_or_else(|| eyre::eyre!("invalid Vulkan model buffer handle {}", handle.0))
    }

    fn model_binding(&self, handle: VulkanModelBufferHandle) -> Result<StorageBufferBinding> {
        let model = self.model_buffer(handle)?;
        Ok(StorageBufferBinding {
            buffer: model.allocation.buffer,
            offset: 0,
            range: model.byte_len,
        })
    }

    fn batch_buffer(&self, handle: VulkanBatchBufferHandle) -> Result<&BatchBufferSlice> {
        self.buffers
            .get(handle.0)
            .ok_or_else(|| eyre::eyre!("invalid Vulkan batch buffer handle {}", handle.0))
    }

    fn batch_allocation(
        &self,
        handle: VulkanBatchBufferHandle,
    ) -> Result<(&BufferAllocation, vk::DeviceSize, vk::DeviceSize)> {
        let slice = self.batch_buffer(handle)?;
        let arena = self.arenas.get(slice.arena_index).ok_or_else(|| {
            eyre::eyre!("invalid Vulkan batch arena handle {}", slice.arena_index)
        })?;
        Ok((&arena.allocation, slice.offset, slice.byte_len))
    }

    fn batch_binding(&self, handle: VulkanBatchBufferHandle) -> Result<StorageBufferBinding> {
        let slice = self.batch_buffer(handle)?;
        let arena = self.arenas.get(slice.arena_index).ok_or_else(|| {
            eyre::eyre!("invalid Vulkan batch arena handle {}", slice.arena_index)
        })?;
        Ok(StorageBufferBinding {
            buffer: arena.allocation.buffer,
            offset: slice.offset,
            range: slice.byte_len,
        })
    }

    fn allocate_descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
        buffers: &[StorageBufferBinding],
    ) -> Result<vk::DescriptorSet> {
        let cache_index = if let Some(index) = self
            .descriptor_sets
            .iter()
            .position(|cache| cache.layout == layout)
        {
            index
        } else {
            self.descriptor_sets.push(DescriptorSetCache {
                layout,
                free: Vec::new(),
            });
            self.descriptor_sets.len() - 1
        };
        if self.descriptor_sets[cache_index].free.is_empty() {
            let layouts = vec![layout; DESCRIPTOR_SET_ALLOCATION_BLOCK];
            let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);
            // SAFETY: The descriptor pool and layouts belong to this live
            // device and the layout slice remains alive for the call.
            let allocated = unsafe {
                self.context
                    .device
                    .allocate_descriptor_sets(&descriptor_set_info)
            }
            .wrap_err("failed to allocate Vulkan model batch descriptor sets")?;
            self.descriptor_sets[cache_index].free.extend(allocated);
        }
        let descriptor_set = self.descriptor_sets[cache_index]
            .free
            .pop()
            .ok_or_else(|| eyre::eyre!("Vulkan returned no model batch descriptor set"))?;
        let infos = buffers
            .iter()
            .map(|binding| {
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.buffer)
                    .offset(binding.offset)
                    .range(binding.range)
            })
            .collect::<Vec<_>>();
        let writes = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(u32::try_from(binding).expect("model batch has few bindings"))
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        // SAFETY: Descriptor handles, layouts, and buffer infos belong to this
        // live device and remain alive through command submission.
        unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
        Ok(descriptor_set)
    }

    fn record_four_buffer_dispatch(
        &mut self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        push_constants: &[u32],
        output_count: usize,
    ) -> Result<()> {
        self.record_dispatch(
            pipeline,
            pipeline_layout,
            descriptor_set,
            push_constants,
            output_count,
        )
    }

    fn record_three_buffer_dispatch(
        &mut self,
        descriptor_set: vk::DescriptorSet,
        push_constants: &[u32],
        output_count: usize,
    ) -> Result<()> {
        self.record_dispatch(
            self.context.elementwise_pipeline.pipeline,
            self.context.elementwise_pipeline.pipeline_layout,
            descriptor_set,
            push_constants,
            output_count,
        )
    }

    fn record_dispatch(
        &mut self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        push_constants: &[u32],
        output_count: usize,
    ) -> Result<()> {
        let workgroups = u32::try_from(output_count.div_ceil(256))
            .wrap_err("Vulkan model batch dispatch dimensions are too large")?;
        self.record_dispatch_with_workgroups(
            pipeline,
            pipeline_layout,
            descriptor_set,
            push_constants,
            workgroups,
        );
        Ok(())
    }

    fn record_dispatch_with_workgroups(
        &mut self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        push_constants: &[u32],
        workgroups: u32,
    ) {
        // SAFETY: The pipeline belongs to this live device.
        unsafe {
            self.context.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
        };
        // SAFETY: The descriptor set layout matches the selected pipeline.
        unsafe {
            self.context.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
        };
        // SAFETY: The push-constant slice views the live u32 array for the
        // duration of this command-recording call.
        let push_constant_bytes = unsafe {
            std::slice::from_raw_parts(
                push_constants.as_ptr().cast::<u8>(),
                std::mem::size_of_val(push_constants),
            )
        };
        // SAFETY: The push constants match the selected pipeline layout.
        unsafe {
            self.context.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constant_bytes,
            );
        };
        // SAFETY: The shaders bounds-check all output invocations.
        unsafe {
            self.context
                .device
                .cmd_dispatch(self.command_buffer, workgroups, 1, 1);
        };
        self.add_compute_barrier();
    }

    fn add_compute_barrier(&mut self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        // SAFETY: The barrier is recorded between compute dispatches in this
        // command buffer, ordering the storage-buffer writes before the next
        // shader reads. The fixed graph gives every dispatch a distinct output
        // slice, so no write-after-write dependency needs to be carried here.
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        };
    }

    fn add_transfer_to_compute_barrier(&mut self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: The barrier is recorded after a staging copy and before the
        // first shader that consumes the copied batch input.
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        };
    }

    fn add_compute_to_transfer_barrier(&mut self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        // SAFETY: The barrier is recorded before output copies and orders all
        // preceding shader writes against those transfer reads.
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        };
    }
}

impl Drop for VulkanBatch<'_> {
    fn drop(&mut self) {
        if self.fence != vk::Fence::null() {
            // SAFETY: The fence belongs to this live device.
            unsafe { self.context.device.destroy_fence(self.fence, None) };
        }
        if self.command_pool != vk::CommandPool::null() {
            // SAFETY: The command pool is no longer needed when the batch is
            // dropped. `finish` has waited for it when it was submitted.
            unsafe {
                self.context
                    .device
                    .destroy_command_pool(self.command_pool, None);
            };
        }
        if self.descriptor_pool != vk::DescriptorPool::null() {
            // SAFETY: Descriptor sets are released with their pool.
            unsafe {
                self.context
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            };
        }
        if let Some(query_pool) = self.timestamp_query_pool {
            // SAFETY: The batch fence was waited before normal destruction.
            unsafe { self.context.device.destroy_query_pool(query_pool, None) };
        }
        for arena in &self.arenas {
            destroy_buffer(&self.context.device, &arena.allocation);
        }
        for staging in &self.staging_buffers {
            destroy_buffer(&self.context.device, staging);
        }
        for readback in &self.readback_buffers {
            destroy_buffer(&self.context.device, readback);
        }
    }
}

impl Drop for VulkanContext {
    #[expect(
        clippy::semicolon_outside_block,
        reason = "Vulkan destruction calls remain grouped with their safety comments."
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "Vulkan pipeline cleanup remains grouped with explicit resource-lifetime comments."
    )]
    fn drop(&mut self) {
        // SAFETY: The context owns the device and all model work has completed
        // before the runtime is dropped.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        if let Some(weights) = self.embedding_weights.take() {
            destroy_buffer(&self.device, &weights.buffer);
        }
        // SAFETY: No submitted work can reference the cached embedding
        // pipeline after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.embedding_pipeline.pipeline, None);
        }
        // SAFETY: The cached embedding pipeline no longer references its layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.embedding_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached embedding pipeline no longer references its shader.
        unsafe {
            self.device
                .destroy_shader_module(self.embedding_pipeline.shader, None);
        }
        // SAFETY: The cached descriptor layout is no longer used by any pool.
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.embedding_pipeline.descriptor_layout, None);
        }
        // SAFETY: No submitted work can reference the cached convolution
        // pipeline after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.conv1d_pipeline.pipeline, None);
        }
        // SAFETY: The cached convolution pipeline no longer references its
        // layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.conv1d_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached convolution pipeline no longer references its
        // shader.
        unsafe {
            self.device
                .destroy_shader_module(self.conv1d_pipeline.shader, None);
        }
        // SAFETY: The cached convolution descriptor layout is no longer used
        // by any pool.
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.conv1d_pipeline.descriptor_layout, None);
        }
        // SAFETY: No submitted work can reference the cached transposed
        // convolution pipeline after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.conv_transpose1d_pipeline.pipeline, None);
        }
        // SAFETY: The cached transposed convolution pipeline no longer
        // references its layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.conv_transpose1d_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached transposed convolution pipeline no longer
        // references its shader.
        unsafe {
            self.device
                .destroy_shader_module(self.conv_transpose1d_pipeline.shader, None);
        }
        // SAFETY: The cached transposed convolution descriptor layout is no
        // longer used by any pool.
        unsafe {
            self.device.destroy_descriptor_set_layout(
                self.conv_transpose1d_pipeline.descriptor_layout,
                None,
            );
        }
        // SAFETY: No submitted work can reference the cached elementwise
        // pipeline after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.elementwise_pipeline.pipeline, None);
        }
        // SAFETY: The cached elementwise pipeline no longer references its
        // layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.elementwise_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached elementwise pipeline no longer references its
        // shader.
        unsafe {
            self.device
                .destroy_shader_module(self.elementwise_pipeline.shader, None);
        }
        // SAFETY: The cached descriptor layout is no longer used by any pool.
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.elementwise_pipeline.descriptor_layout, None);
        }
        // SAFETY: No submitted work can reference the cached linear pipeline
        // after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.linear_pipeline.pipeline, None);
        }
        // SAFETY: The cached linear pipeline no longer references its layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.linear_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached linear pipeline no longer references its shader.
        unsafe {
            self.device
                .destroy_shader_module(self.linear_pipeline.shader, None);
        }
        // SAFETY: The cached linear descriptor layout is no longer used by
        // any pool.
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.linear_pipeline.descriptor_layout, None);
        }
        // SAFETY: No submitted work can reference the cached length-regulation
        // pipeline after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.length_regulate_pipeline.pipeline, None);
        }
        // SAFETY: The cached length-regulation pipeline no longer references
        // its layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.length_regulate_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached length-regulation pipeline no longer references
        // its shader.
        unsafe {
            self.device
                .destroy_shader_module(self.length_regulate_pipeline.shader, None);
        }
        // SAFETY: The cached descriptor layout is no longer used by any pool.
        unsafe {
            self.device.destroy_descriptor_set_layout(
                self.length_regulate_pipeline.descriptor_layout,
                None,
            );
        }
        // SAFETY: No submitted work can reference the cached LSTM pipeline
        // after the device-idle wait.
        unsafe {
            self.device
                .destroy_pipeline(self.lstm_pipeline.pipeline, None);
        }
        // SAFETY: The cached LSTM pipeline no longer references its layout.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.lstm_pipeline.pipeline_layout, None);
        }
        // SAFETY: The cached LSTM pipeline no longer references its shader.
        unsafe {
            self.device
                .destroy_shader_module(self.lstm_pipeline.shader, None);
        }
        // SAFETY: The cached LSTM descriptor layout is no longer used by any
        // pool.
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.lstm_pipeline.descriptor_layout, None);
        }
        destroy_simple_compute_pipeline(&self.device, &self.batch_norm_pipeline);
        destroy_simple_compute_pipeline(&self.device, &self.max_pool1d_pipeline);
        destroy_simple_compute_pipeline(&self.device, &self.gru_pipeline);
        destroy_simple_compute_pipeline(&self.device, &self.copy_channels_pipeline);
        for model_buffer in self.model_buffers.drain(..) {
            destroy_buffer(&self.device, &model_buffer.allocation);
        }
        // SAFETY: All child resources owned by the context have been dropped
        // before its logical device.
        unsafe { self.device.destroy_device(None) };
        // SAFETY: The instance outlives the logical device and is owned here.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn destroy_simple_compute_pipeline(device: &ash::Device, pipeline: &SimpleComputePipeline) {
    // SAFETY: The context waited for device idle before destroying child
    // resources, and this helper is called before the logical device drops.
    unsafe { device.destroy_pipeline(pipeline.pipeline, None) };
    // SAFETY: The pipeline no longer references its layout.
    unsafe { device.destroy_pipeline_layout(pipeline.pipeline_layout, None) };
    // SAFETY: The pipeline no longer references its shader module.
    unsafe { device.destroy_shader_module(pipeline.shader, None) };
    // SAFETY: No descriptor pool uses this layout after the device-idle wait.
    unsafe { device.destroy_descriptor_set_layout(pipeline.descriptor_layout, None) };
}

fn create_embedding_pipeline(device: &ash::Device) -> Result<EmbeddingPipeline> {
    let shader_words = spirv_words(EMBEDDING_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan embedding shader module")?;
    let descriptor_bindings = [storage_binding(0), storage_binding(1), storage_binding(2)];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor binding data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan embedding descriptor layout")?;
    let push_constant_size =
        u32::try_from(std::mem::size_of::<[u32; 2]>()).expect("embedding push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan embedding pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create Vulkan embedding pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no embedding pipeline"))?;
    Ok(EmbeddingPipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_conv1d_pipeline(device: &ash::Device) -> Result<Conv1dPipeline> {
    let shader_words = spirv_words(CONV1D_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan Conv1d shader module")?;
    let descriptor_bindings = [
        storage_binding(0),
        storage_binding(1),
        storage_binding(2),
        storage_binding(3),
    ];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor binding data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan Conv1d descriptor layout")?;
    let push_constant_size =
        u32::try_from(std::mem::size_of::<[u32; 10]>()).expect("Conv1d push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan Conv1d pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create Vulkan Conv1d pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no Conv1d pipeline"))?;
    Ok(Conv1dPipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_conv_transpose1d_pipeline(device: &ash::Device) -> Result<ConvTranspose1dPipeline> {
    let shader_words = spirv_words(CONV_TRANSPOSE1D_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan ConvTranspose1d shader module")?;
    let descriptor_bindings = [
        storage_binding(0),
        storage_binding(1),
        storage_binding(2),
        storage_binding(3),
    ];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor binding data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan ConvTranspose1d descriptor layout")?;
    let push_constant_size = u32::try_from(std::mem::size_of::<[u32; 10]>())
        .expect("ConvTranspose1d push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan ConvTranspose1d pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| {
        eyre::eyre!("failed to create Vulkan ConvTranspose1d pipeline: {error:?}")
    })?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no ConvTranspose1d pipeline"))?;
    Ok(ConvTranspose1dPipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_elementwise_pipeline(device: &ash::Device) -> Result<ElementwisePipeline> {
    let shader_words = spirv_words(ELEMENTWISE_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan elementwise shader module")?;
    let descriptor_bindings = [storage_binding(0), storage_binding(1), storage_binding(2)];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor layout data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan elementwise descriptor layout")?;
    let push_constant_size =
        u32::try_from(std::mem::size_of::<[u32; 3]>()).expect("elementwise push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan elementwise pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create Vulkan elementwise pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no elementwise pipeline"))?;
    Ok(ElementwisePipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_linear_pipeline(device: &ash::Device) -> Result<LinearPipeline> {
    let shader_words = spirv_words(LINEAR_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan linear shader module")?;
    let descriptor_bindings = [
        storage_binding(0),
        storage_binding(1),
        storage_binding(2),
        storage_binding(3),
    ];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor layout data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan linear descriptor layout")?;
    let push_constant_size =
        u32::try_from(std::mem::size_of::<[u32; 4]>()).expect("linear push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan linear pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create Vulkan linear pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no linear pipeline"))?;
    Ok(LinearPipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_length_regulate_pipeline(device: &ash::Device) -> Result<LengthRegulatePipeline> {
    let shader_words = spirv_words(LENGTH_REGULATE_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan length-regulation shader module")?;
    let descriptor_bindings = [storage_binding(0), storage_binding(1), storage_binding(2)];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor layout data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan length-regulation descriptor layout")?;
    let push_constant_size = u32::try_from(std::mem::size_of::<[u32; 3]>())
        .expect("length-regulation push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan length-regulation pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| {
        eyre::eyre!("failed to create Vulkan length-regulation pipeline: {error:?}")
    })?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no length-regulation pipeline"))?;
    Ok(LengthRegulatePipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_lstm_pipeline(device: &ash::Device) -> Result<LstmPipeline> {
    let shader_words = spirv_words(LSTM_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the Vulkan LSTM shader module")?;
    let descriptor_bindings = [
        storage_binding(0),
        storage_binding(1),
        storage_binding(2),
        storage_binding(3),
        storage_binding(4),
    ];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor layout data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err("failed to create the Vulkan LSTM descriptor layout")?;
    let push_constant_size =
        u32::try_from(std::mem::size_of::<[u32; 4]>()).expect("LSTM push constants fit u32");
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err("failed to create the Vulkan LSTM pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create Vulkan LSTM pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no LSTM pipeline"))?;
    Ok(LstmPipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

fn create_simple_compute_pipeline(
    device: &ash::Device,
    shader_bytes: &[u8],
    binding_count: usize,
    push_constant_size: usize,
    operation_name: &str,
) -> Result<SimpleComputePipeline> {
    let shader_words = spirv_words(shader_bytes)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device.create_shader_module(&shader_info, None) }
        .wrap_err_with(|| format!("failed to create the Vulkan {operation_name} shader module"))?;
    let descriptor_bindings = (0..binding_count)
        .map(|binding| storage_binding(u32::try_from(binding).expect("Vulkan has few bindings")))
        .collect::<Vec<_>>();
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor binding data remains alive for the call.
    let descriptor_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }
            .wrap_err_with(|| {
                format!("failed to create the Vulkan {operation_name} descriptor layout")
            })?;
    let push_constant_size =
        u32::try_from(push_constant_size).wrap_err("Vulkan push-constant range is too large")?;
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .wrap_err_with(|| {
            format!("failed to create the Vulkan {operation_name} pipeline layout")
        })?;
    let entry_name = CString::new("main").expect("static shader entry has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| {
        eyre::eyre!("failed to create Vulkan {operation_name} pipeline: {error:?}")
    })?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no {operation_name} pipeline"))?;
    Ok(SimpleComputePipeline {
        shader,
        descriptor_layout,
        pipeline_layout,
        pipeline,
    })
}

/// Report for one enumerated physical device.
#[derive(Clone, Debug, Facet, PartialEq)]
pub struct VulkanDeviceReport {
    pub name: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: String,
    pub driver_version: u32,
    pub compute_queue_family_index: Option<u32>,
    pub compute_timestamp_valid_bits: Option<u32>,
    pub cooperative_matrix_khr: bool,
    pub cooperative_matrix_properties: Vec<String>,
}

/// Report from the optional Vulkan capability probe.
#[expect(
    clippy::struct_excessive_bools,
    reason = "The JSON probe report deliberately exposes independent capability facts."
)]
#[derive(Clone, Debug, Facet, PartialEq)]
pub struct VulkanProbeReport {
    pub instance_api_version: String,
    pub physical_device_count: usize,
    pub selected_device_index: Option<usize>,
    pub selected_device_name: Option<String>,
    pub selected_is_rtx_4090: bool,
    pub selected_compute_queue_family_index: Option<u32>,
    pub selected_cooperative_matrix_khr: bool,
    pub selected_cooperative_matrix_properties: Vec<String>,
    pub vector_add_passed: bool,
    pub vector_add_elapsed_ns: Option<u64>,
    pub vector_add_max_abs_error: Option<f32>,
    pub vector_add_error: Option<String>,
    pub matrix_multiply_passed: bool,
    pub matrix_multiply_elapsed_ns: Option<u64>,
    pub matrix_multiply_gpu_elapsed_ns: Option<u64>,
    pub matrix_multiply_max_abs_error: Option<f32>,
    pub matrix_multiply_error: Option<String>,
    pub embedding_passed: bool,
    pub embedding_cold_elapsed_ns: Option<u64>,
    pub embedding_warm_elapsed_ns: Option<u64>,
    pub embedding_max_abs_error: Option<f32>,
    pub embedding_error: Option<String>,
    pub conv1d_passed: bool,
    pub conv1d_elapsed_ns: Option<u64>,
    pub conv1d_max_abs_error: Option<f32>,
    pub conv1d_error: Option<String>,
    pub conv_transpose1d_passed: bool,
    pub conv_transpose1d_elapsed_ns: Option<u64>,
    pub conv_transpose1d_max_abs_error: Option<f32>,
    pub conv_transpose1d_error: Option<String>,
    pub devices: Vec<VulkanDeviceReport>,
}

/// Enumerate Vulkan devices and report the capability facts needed by W17.
///
/// # Errors
///
/// Returns an error when the Vulkan loader or instance cannot be loaded, or
/// when physical-device enumeration fails.
///
/// # Panics
///
/// Panics only if the fixed synthetic embedding probe exceeds the integer
/// widths used by its test fixture.
#[expect(
    clippy::too_many_lines,
    reason = "The probe keeps device selection and its evidence report together."
)]
pub fn probe() -> Result<VulkanProbeReport> {
    // SAFETY: Entry::load performs the platform Vulkan-loader lookup and
    // returns an owned entry whose function pointers remain valid while it is
    // alive.
    let entry = unsafe { Entry::load() }.wrap_err(
        "failed to load the Vulkan loader; install a Vulkan driver/runtime before probing",
    )?;
    // SAFETY: Querying the loader version does not require an instance.
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .wrap_err("failed to query the Vulkan loader version")?
        .unwrap_or(vk::API_VERSION_1_0)
        .min(vk::API_VERSION_1_3);

    let application_name =
        CString::new("teamy-tts").wrap_err("failed to build Vulkan application name")?;
    let engine_name =
        CString::new("teamy-tts-vulkan-probe").wrap_err("failed to build Vulkan engine name")?;
    let application_info = vk::ApplicationInfo::default()
        .application_name(&application_name)
        .application_version(1)
        .engine_name(&engine_name)
        .engine_version(1)
        .api_version(loader_version);
    let create_info = vk::InstanceCreateInfo::default().application_info(&application_info);
    // SAFETY: create_info points to strings and structures alive for this
    // call; no extensions or layers are requested.
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .wrap_err("failed to create a Vulkan instance")?;
    let guard = InstanceGuard { instance };

    // SAFETY: The instance is alive and Vulkan owns the returned physical
    // device handles for the guard's lifetime.
    let physical_devices = unsafe { guard.instance.enumerate_physical_devices() }
        .wrap_err("failed to enumerate Vulkan physical devices")?;
    if physical_devices.is_empty() {
        bail!("Vulkan loader is present but no physical devices were enumerated");
    }

    let mut devices = Vec::with_capacity(physical_devices.len());
    let mut selected_index = None;
    for (index, physical_device) in physical_devices.iter().copied().enumerate() {
        let report = inspect_device(&entry, &guard.instance, physical_device)?;
        let is_preferred = report.compute_queue_family_index.is_some()
            && (report.device_type == "DISCRETE_GPU" || report.name.contains("RTX 4090"));
        if is_preferred && selected_index.is_none() {
            selected_index = Some(index);
        }
        devices.push(report);
    }

    let selected = selected_index.or_else(|| {
        devices
            .iter()
            .enumerate()
            .find(|(_, device)| device.compute_queue_family_index.is_some())
            .map(|(index, _)| index)
    });
    let (selected_device_name, selected_is_rtx_4090, selected_queue, selected_cooperative, props) =
        selected.map_or_else(
            || (None, false, None, false, Vec::new()),
            |index| {
                let device = &devices[index];
                (
                    Some(device.name.clone()),
                    device.name.contains("RTX 4090"),
                    device.compute_queue_family_index,
                    device.cooperative_matrix_khr,
                    device.cooperative_matrix_properties.clone(),
                )
            },
        );
    let mut vector_add_passed = false;
    let mut vector_add_elapsed_ns = None;
    let mut vector_add_max_abs_error = None;
    let mut vector_add_error = None;
    let mut matrix_multiply_passed = false;
    let mut matrix_multiply_elapsed_ns = None;
    let mut matrix_multiply_gpu_elapsed_ns = None;
    let mut matrix_multiply_max_abs_error = None;
    let mut matrix_multiply_error = None;
    let mut embedding_passed = false;
    let mut embedding_cold_elapsed_ns = None;
    let mut embedding_warm_elapsed_ns = None;
    let mut embedding_max_abs_error = None;
    let mut embedding_error = None;
    let mut conv1d_passed = false;
    let mut conv1d_elapsed_ns = None;
    let mut conv1d_max_abs_error = None;
    let mut conv1d_error = None;
    let mut conv_transpose1d_passed = false;
    let mut conv_transpose1d_elapsed_ns = None;
    let mut conv_transpose1d_max_abs_error = None;
    let mut conv_transpose1d_error = None;
    if let (Some(index), Some(queue_family_index)) = (selected, selected_queue) {
        match run_vector_add(&guard.instance, physical_devices[index], queue_family_index) {
            Ok(result) => {
                vector_add_passed = result.passed;
                vector_add_elapsed_ns = Some(result.elapsed_ns);
                vector_add_max_abs_error = Some(result.max_abs_error);
            }
            Err(error) => {
                vector_add_error = Some(error.to_string());
            }
        }
        // SAFETY: The physical device handle came from this live instance.
        let timestamp_period_ns = unsafe {
            guard
                .instance
                .get_physical_device_properties(physical_devices[index])
        }
        .limits
        .timestamp_period;
        let timestamp_valid_bits = devices[index]
            .compute_timestamp_valid_bits
            .unwrap_or_default();
        match run_matrix_multiply(
            &guard.instance,
            physical_devices[index],
            queue_family_index,
            timestamp_period_ns,
            timestamp_valid_bits,
        ) {
            Ok(result) => {
                matrix_multiply_passed = result.passed;
                matrix_multiply_elapsed_ns = Some(result.elapsed_ns);
                matrix_multiply_gpu_elapsed_ns = result.gpu_elapsed_ns;
                matrix_multiply_max_abs_error = Some(result.max_abs_error);
            }
            Err(error) => {
                matrix_multiply_error = Some(error.to_string());
            }
        }
        let vocabulary_size = 32_usize;
        let embedding_dimension = 16_usize;
        let embedding_weights = (0..vocabulary_size * embedding_dimension)
            .map(|index| {
                f32::from(u16::try_from(index % 17).expect("embedding fixture fits u16")) * 0.0625
            })
            .collect::<Vec<_>>();
        let embedding_tokens = vec![1_u32, 3, 7, 15];
        match VulkanContext::new() {
            Ok(mut context) => {
                if let Err(error) = context.prepare_embedding(
                    &embedding_weights,
                    vocabulary_size,
                    embedding_dimension,
                ) {
                    embedding_error = Some(error.to_string());
                    return Ok(VulkanProbeReport {
                        instance_api_version: format_version(loader_version),
                        physical_device_count: physical_devices.len(),
                        selected_device_index: selected,
                        selected_device_name,
                        selected_is_rtx_4090,
                        selected_compute_queue_family_index: selected_queue,
                        selected_cooperative_matrix_khr: selected_cooperative,
                        selected_cooperative_matrix_properties: props,
                        vector_add_passed,
                        vector_add_elapsed_ns,
                        vector_add_max_abs_error,
                        vector_add_error,
                        matrix_multiply_passed,
                        matrix_multiply_elapsed_ns,
                        matrix_multiply_gpu_elapsed_ns,
                        matrix_multiply_max_abs_error,
                        matrix_multiply_error,
                        embedding_passed,
                        embedding_cold_elapsed_ns,
                        embedding_warm_elapsed_ns,
                        embedding_max_abs_error,
                        embedding_error,
                        conv1d_passed,
                        conv1d_elapsed_ns,
                        conv1d_max_abs_error,
                        conv1d_error,
                        conv_transpose1d_passed,
                        conv_transpose1d_elapsed_ns,
                        conv_transpose1d_max_abs_error,
                        conv_transpose1d_error,
                        devices,
                    });
                }
                let conv_input_channels = 2_usize;
                let conv_input_length = 5_usize;
                let conv_output_channels = 3_usize;
                let conv_kernel_size = 3_usize;
                let conv_weights = (0..conv_output_channels
                    * conv_input_channels
                    * conv_kernel_size)
                    .map(|index| {
                        f32::from(u16::try_from(index).expect("Conv1d fixture fits u16")) * 0.03125
                    })
                    .collect::<Vec<_>>();
                let conv_bias = vec![0.125_f32, -0.25, 0.5];
                let conv_input = (0..conv_input_channels * conv_input_length)
                    .map(|index| {
                        f32::from(u16::try_from(index + 1).expect("Conv1d fixture fits u16"))
                            * 0.0625
                    })
                    .collect::<Vec<_>>();
                let conv_started = std::time::Instant::now();
                let conv_result = context.dispatch_conv1d(
                    &conv_weights,
                    conv_output_channels,
                    conv_input_channels,
                    conv_kernel_size,
                    Some(&conv_bias),
                    &conv_input,
                    conv_input_length,
                    1,
                    1,
                    1,
                    1,
                );
                conv1d_elapsed_ns =
                    Some(u64::try_from(conv_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                match conv_result {
                    Ok(output) => {
                        let expected = (0..conv_output_channels * conv_input_length)
                            .map(|linear_index| {
                                let output_channel = linear_index / conv_input_length;
                                let output_position = linear_index % conv_input_length;
                                let mut value = conv_bias[output_channel];
                                for input_channel in 0..conv_input_channels {
                                    for kernel_position in 0..conv_kernel_size {
                                        let input_position = output_position.cast_signed()
                                            + kernel_position.cast_signed()
                                            - 1;
                                        if (0..conv_input_length.cast_signed())
                                            .contains(&input_position)
                                        {
                                            let input_index = input_channel * conv_input_length
                                                + usize::try_from(input_position).expect(
                                                    "fixture input position is non-negative",
                                                );
                                            let weight_index = (output_channel
                                                * conv_input_channels
                                                + input_channel)
                                                * conv_kernel_size
                                                + kernel_position;
                                            value += conv_input[input_index]
                                                * conv_weights[weight_index];
                                        }
                                    }
                                }
                                value
                            })
                            .collect::<Vec<_>>();
                        let max_abs_error = output
                            .iter()
                            .zip(&expected)
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f32, f32::max);
                        conv1d_max_abs_error = Some(max_abs_error);
                        conv1d_passed = max_abs_error <= 1.0e-5;
                    }
                    Err(error) => {
                        conv1d_error = Some(error.to_string());
                    }
                }
                let transpose_input_channels = 2_usize;
                let transpose_input_length = 3_usize;
                let transpose_output_channels = 2_usize;
                let transpose_kernel_size = 4_usize;
                let transpose_weights = (0..transpose_input_channels
                    * transpose_output_channels
                    * transpose_kernel_size)
                    .map(|index| {
                        f32::from(u16::try_from(index).expect("ConvTranspose1d fixture fits u16"))
                            * 0.03125
                    })
                    .collect::<Vec<_>>();
                let transpose_bias = vec![0.25_f32, -0.125];
                let transpose_input = (0..transpose_input_channels * transpose_input_length)
                    .map(|index| {
                        f32::from(
                            u16::try_from(index + 1).expect("ConvTranspose1d fixture fits u16"),
                        ) * 0.0625
                    })
                    .collect::<Vec<_>>();
                let transpose_started = std::time::Instant::now();
                let transpose_result = context.dispatch_conv_transpose1d(
                    &transpose_weights,
                    transpose_input_channels,
                    transpose_output_channels,
                    transpose_kernel_size,
                    Some(&transpose_bias),
                    &transpose_input,
                    transpose_input_length,
                    2,
                    1,
                    1,
                    0,
                );
                conv_transpose1d_elapsed_ns =
                    Some(u64::try_from(transpose_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                match transpose_result {
                    Ok(output) => {
                        let output_length = 6_usize;
                        let expected = (0..transpose_output_channels * output_length)
                            .map(|linear_index| {
                                let output_channel = linear_index / output_length;
                                let output_position = linear_index % output_length;
                                let mut value = transpose_bias[output_channel];
                                for input_channel in 0..transpose_input_channels {
                                    for kernel_position in 0..transpose_kernel_size {
                                        let numerator = output_position.cast_signed() + 1
                                            - kernel_position.cast_signed();
                                        if numerator >= 0 && numerator % 2 == 0 {
                                            let input_position = usize::try_from(numerator / 2)
                                                .expect("fixture input position is non-negative");
                                            if input_position < transpose_input_length {
                                                let input_index = input_channel
                                                    * transpose_input_length
                                                    + input_position;
                                                let weight_index = (input_channel
                                                    * transpose_output_channels
                                                    + output_channel)
                                                    * transpose_kernel_size
                                                    + kernel_position;
                                                value += transpose_input[input_index]
                                                    * transpose_weights[weight_index];
                                            }
                                        }
                                    }
                                }
                                value
                            })
                            .collect::<Vec<_>>();
                        let max_abs_error = output
                            .iter()
                            .zip(&expected)
                            .map(|(actual, expected)| (actual - expected).abs())
                            .fold(0.0_f32, f32::max);
                        conv_transpose1d_max_abs_error = Some(max_abs_error);
                        conv_transpose1d_passed = max_abs_error <= 1.0e-5;
                    }
                    Err(error) => {
                        conv_transpose1d_error = Some(error.to_string());
                    }
                }
                let cold_started = std::time::Instant::now();
                let cold_result = context.dispatch_prepared_embedding(&embedding_tokens);
                embedding_cold_elapsed_ns =
                    Some(u64::try_from(cold_started.elapsed().as_nanos()).unwrap_or(u64::MAX));
                match cold_result {
                    Ok(cold_output) => {
                        let warm_started = std::time::Instant::now();
                        match context.dispatch_prepared_embedding(&embedding_tokens) {
                            Ok(output) => {
                                embedding_warm_elapsed_ns = Some(
                                    u64::try_from(warm_started.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX),
                                );
                                let max_abs_error = output
                                    .iter()
                                    .enumerate()
                                    .map(|(index, &value)| {
                                        let token = usize::try_from(
                                            embedding_tokens[index / embedding_dimension],
                                        )
                                        .expect("embedding fixture token fits usize");
                                        let expected = embedding_weights[token
                                            * embedding_dimension
                                            + index % embedding_dimension];
                                        (value - expected).abs()
                                    })
                                    .fold(0.0_f32, f32::max);
                                let repeat_error = output
                                    .iter()
                                    .zip(&cold_output)
                                    .map(|(&value, &cold)| (value - cold).abs())
                                    .fold(0.0_f32, f32::max);
                                embedding_max_abs_error = Some(max_abs_error.max(repeat_error));
                                embedding_passed =
                                    max_abs_error <= 1.0e-6 && repeat_error <= 1.0e-6;
                            }
                            Err(error) => {
                                embedding_error = Some(error.to_string());
                            }
                        }
                    }
                    Err(error) => {
                        embedding_error = Some(error.to_string());
                    }
                }
            }
            Err(error) => {
                embedding_error = Some(error.to_string());
            }
        }
    } else {
        vector_add_error = Some("no compute-capable Vulkan queue was found".to_string());
        matrix_multiply_error = Some("no compute-capable Vulkan queue was found".to_string());
        embedding_error = Some("no compute-capable Vulkan queue was found".to_string());
        conv1d_error = Some("no compute-capable Vulkan queue was found".to_string());
        conv_transpose1d_error = Some("no compute-capable Vulkan queue was found".to_string());
    }

    Ok(VulkanProbeReport {
        instance_api_version: format_version(loader_version),
        physical_device_count: physical_devices.len(),
        selected_device_index: selected,
        selected_device_name,
        selected_is_rtx_4090,
        selected_compute_queue_family_index: selected_queue,
        selected_cooperative_matrix_khr: selected_cooperative,
        selected_cooperative_matrix_properties: props,
        vector_add_passed,
        vector_add_elapsed_ns,
        vector_add_max_abs_error,
        vector_add_error,
        matrix_multiply_passed,
        matrix_multiply_elapsed_ns,
        matrix_multiply_gpu_elapsed_ns,
        matrix_multiply_max_abs_error,
        matrix_multiply_error,
        embedding_passed,
        embedding_cold_elapsed_ns,
        embedding_warm_elapsed_ns,
        embedding_max_abs_error,
        embedding_error,
        conv1d_passed,
        conv1d_elapsed_ns,
        conv1d_max_abs_error,
        conv1d_error,
        conv_transpose1d_passed,
        conv_transpose1d_elapsed_ns,
        conv_transpose1d_max_abs_error,
        conv_transpose1d_error,
        devices,
    })
}

fn inspect_device(
    entry: &Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<VulkanDeviceReport> {
    // SAFETY: physical_device came from this live instance.
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let name = properties.device_name_as_c_str().map_or_else(
        |_| "unnamed Vulkan device".to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    // SAFETY: physical_device came from this live instance.
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let compute_queue_family_index = queue_families
        .iter()
        .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .and_then(|index| u32::try_from(index).ok());
    let compute_timestamp_valid_bits = compute_queue_family_index
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| queue_families.get(index))
        .map(|family| family.timestamp_valid_bits);
    // SAFETY: physical_device came from this live instance.
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .wrap_err_with(|| format!("failed to enumerate extensions for {name}"))?;
    let cooperative_matrix_khr = extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .is_ok_and(|extension| extension == ash::khr::cooperative_matrix::NAME)
    });
    let cooperative_matrix_properties = if cooperative_matrix_khr {
        let extension = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        // SAFETY: The device advertises VK_KHR_cooperative_matrix and the
        // extension function was loaded from this live instance.
        let properties =
            unsafe { extension.get_physical_device_cooperative_matrix_properties(physical_device) }
                .wrap_err_with(|| {
                    format!("failed to query cooperative-matrix properties for {name}")
                })?;
        properties
            .iter()
            .map(|property| {
                format!(
                    "m={} n={} k={} a={:?} b={:?} c={:?} result={:?} scope={:?}",
                    property.m_size,
                    property.n_size,
                    property.k_size,
                    property.a_type,
                    property.b_type,
                    property.c_type,
                    property.result_type,
                    property.scope
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(VulkanDeviceReport {
        name,
        device_type: format!("{:?}", properties.device_type),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        api_version: format_version(properties.api_version),
        driver_version: properties.driver_version,
        compute_queue_family_index,
        compute_timestamp_valid_bits,
        cooperative_matrix_khr,
        cooperative_matrix_properties,
    })
}

fn format_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

struct InstanceGuard {
    instance: ash::Instance,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: This guard owns the instance and all child queries have
        // completed before destruction.
        unsafe { self.instance.destroy_instance(None) };
    }
}

struct VectorAddResult {
    passed: bool,
    elapsed_ns: u64,
    gpu_elapsed_ns: Option<u64>,
    max_abs_error: f32,
}

struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

#[expect(
    clippy::too_many_lines,
    reason = "The first Vulkan probe keeps resource creation, dispatch, validation, and cleanup auditable."
)]
#[expect(
    clippy::semicolon_outside_block,
    reason = "Ash command calls are kept in explicit unsafe blocks with safety comments."
)]
#[expect(
    clippy::semicolon_if_nothing_returned,
    reason = "Ash destruction calls are kept in explicit unsafe blocks with safety comments."
)]
fn run_vector_add(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<VectorAddResult> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);
    let device_info =
        vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
    // SAFETY: The queue family was reported by this physical device and the
    // create-info data remains alive for the call.
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .wrap_err("failed to create the Vulkan compute device")?;
    let device_guard = DeviceGuard { device };
    // SAFETY: The queue family and queue index were validated during probe.
    let queue = unsafe { device_guard.device.get_device_queue(queue_family_index, 0) };
    let memory_properties =
        // SAFETY: The physical device belongs to the live instance.
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let a = (0..VECTOR_ADD_ELEMENT_COUNT)
        .map(|index| f32::from(u16::try_from(index).expect("vector-add fixture fits u16")))
        .collect::<Vec<_>>();
    let b = vec![0.25_f32; VECTOR_ADD_ELEMENT_COUNT];
    let a_buffer = create_buffer(
        &device_guard.device,
        &memory_properties,
        VECTOR_ADD_BUFFER_SIZE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let b_buffer = create_buffer(
        &device_guard.device,
        &memory_properties,
        VECTOR_ADD_BUFFER_SIZE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let output_buffer = create_buffer(
        &device_guard.device,
        &memory_properties,
        VECTOR_ADD_BUFFER_SIZE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    write_buffer(&device_guard.device, &a_buffer, &a, VECTOR_ADD_BUFFER_SIZE)?;
    write_buffer(&device_guard.device, &b_buffer, &b, VECTOR_ADD_BUFFER_SIZE)?;

    let shader_words = spirv_words(VECTOR_ADD_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device_guard.device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the vector-add shader module")?;
    let descriptor_bindings = [storage_binding(0), storage_binding(1), storage_binding(2)];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor binding data remains alive for the call.
    let descriptor_layout = unsafe {
        device_guard
            .device
            .create_descriptor_set_layout(&descriptor_layout_info, None)
    }
    .wrap_err("failed to create the vector-add descriptor layout")?;
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe {
        device_guard
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    }
    .wrap_err("failed to create the vector-add pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry name has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device_guard.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create vector-add pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no vector-add pipeline"))?;
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3);
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(std::slice::from_ref(&pool_size));
    // SAFETY: Descriptor pool data remains alive for the call.
    let descriptor_pool = unsafe {
        device_guard
            .device
            .create_descriptor_pool(&descriptor_pool_info, None)
    }
    .wrap_err("failed to create the vector-add descriptor pool")?;
    let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_layout));
    // SAFETY: The pool and layout belong to this live device.
    let descriptor_set = unsafe {
        device_guard
            .device
            .allocate_descriptor_sets(&descriptor_set_info)
    }
    .wrap_err("failed to allocate the vector-add descriptor set")?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no vector-add descriptor set"))?;
    let a_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(a_buffer.buffer)
        .offset(0)
        .range(VECTOR_ADD_BUFFER_SIZE);
    let b_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(b_buffer.buffer)
        .offset(0)
        .range(VECTOR_ADD_BUFFER_SIZE);
    let output_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(output_buffer.buffer)
        .offset(0)
        .range(VECTOR_ADD_BUFFER_SIZE);
    let descriptor_writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&a_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&b_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&output_descriptor)),
    ];
    // SAFETY: Descriptor handles and buffer infos belong to this live device.
    unsafe {
        device_guard
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }
    let command_pool_info =
        vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
    // SAFETY: The queue family was selected from this physical device.
    let command_pool = unsafe {
        device_guard
            .device
            .create_command_pool(&command_pool_info, None)
    }
    .wrap_err("failed to create the vector-add command pool")?;
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: The command pool belongs to this live device.
    let command_buffer = unsafe {
        device_guard
            .device
            .allocate_command_buffers(&command_buffer_info)
    }
    .wrap_err("failed to allocate the vector-add command buffer")?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no vector-add command buffer"))?;
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: The command buffer was allocated from this live device.
    unsafe {
        device_guard
            .device
            .begin_command_buffer(command_buffer, &begin_info)
    }
    .wrap_err("failed to begin the vector-add command buffer")?;
    // SAFETY: The pipeline and descriptor set belong to this live device.
    unsafe {
        device_guard.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline,
        );
    }
    // SAFETY: The descriptor set layout matches the compute shader bindings.
    unsafe {
        device_guard.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        );
    }
    // SAFETY: Four workgroups cover exactly the fixed-size probe buffers.
    unsafe {
        device_guard
            .device
            .cmd_dispatch(command_buffer, VECTOR_ADD_WORKGROUP_COUNT, 1, 1);
    }
    // SAFETY: The command buffer recording has a valid begin/dispatch sequence.
    unsafe { device_guard.device.end_command_buffer(command_buffer) }
        .wrap_err("failed to end the vector-add command buffer")?;
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    // SAFETY: The queue and command buffer belong to this live device.
    let fence = unsafe {
        device_guard
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .wrap_err("failed to create the vector-add fence")?;
    let started = std::time::Instant::now();
    // SAFETY: The submit info references the recorded command buffer.
    unsafe {
        device_guard
            .device
            .queue_submit(queue, std::slice::from_ref(&submit_info), fence)
    }
    .wrap_err("failed to submit the vector-add command buffer")?;
    // SAFETY: The fence belongs to this live device and was signaled by the
    // submitted command buffer.
    unsafe {
        device_guard
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
    }
    .wrap_err("failed while waiting for the vector-add fence")?;
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let output = read_buffer(
        &device_guard.device,
        &output_buffer,
        VECTOR_ADD_BUFFER_SIZE,
        VECTOR_ADD_ELEMENT_COUNT,
    )?;
    let max_abs_error = output
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let expected = a[index] + b[index];
            (value - expected).abs()
        })
        .fold(0.0_f32, f32::max);
    let passed = max_abs_error <= 1.0e-5;

    // SAFETY: All handles below belong to the live device and are no longer
    // referenced by in-flight work after the fence wait.
    unsafe { device_guard.device.destroy_fence(fence, None) };
    // SAFETY: The command pool is no longer in use after the fence wait.
    unsafe { device_guard.device.destroy_command_pool(command_pool, None) };
    // SAFETY: Descriptor objects are no longer used by submitted work.
    unsafe {
        device_guard
            .device
            .destroy_descriptor_pool(descriptor_pool, None)
    };
    // SAFETY: The pipeline and its layout are no longer referenced.
    unsafe { device_guard.device.destroy_pipeline(pipeline, None) };
    // SAFETY: The pipeline layout is no longer referenced.
    unsafe {
        device_guard
            .device
            .destroy_pipeline_layout(pipeline_layout, None)
    };
    // SAFETY: The shader module is no longer referenced by the pipeline.
    unsafe { device_guard.device.destroy_shader_module(shader, None) };
    // SAFETY: The descriptor layout is no longer referenced.
    unsafe {
        device_guard
            .device
            .destroy_descriptor_set_layout(descriptor_layout, None)
    };
    destroy_buffer(&device_guard.device, &a_buffer);
    destroy_buffer(&device_guard.device, &b_buffer);
    destroy_buffer(&device_guard.device, &output_buffer);

    Ok(VectorAddResult {
        passed,
        elapsed_ns,
        gpu_elapsed_ns: None,
        max_abs_error,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "The matrix probe keeps fixed-shape resource setup and validation auditable."
)]
#[expect(
    clippy::semicolon_outside_block,
    reason = "Ash command calls are kept in explicit unsafe blocks with safety comments."
)]
#[expect(
    clippy::semicolon_if_nothing_returned,
    reason = "Ash destruction calls are kept in explicit unsafe blocks with safety comments."
)]
fn run_matrix_multiply(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    timestamp_period_ns: f32,
    timestamp_valid_bits: u32,
) -> Result<VectorAddResult> {
    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);
    let device_info =
        vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
    // SAFETY: The queue family was reported by this physical device and the
    // create-info data remains alive for the call.
    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .wrap_err("failed to create the Vulkan matrix-multiply device")?;
    let device_guard = DeviceGuard { device };
    // SAFETY: The queue family and queue index were validated during probe.
    let queue = unsafe { device_guard.device.get_device_queue(queue_family_index, 0) };
    let memory_properties =
        // SAFETY: The physical device belongs to the live instance.
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let a = (0..MATMUL_ELEMENT_COUNT)
        .map(|index| {
            (f32::from(u16::try_from(index % MATMUL_DIMENSION).expect("matrix fixture fits u16"))
                + 1.0)
                * 0.03125
        })
        .collect::<Vec<_>>();
    let b = (0..MATMUL_ELEMENT_COUNT)
        .map(|index| {
            (f32::from(u16::try_from(index / MATMUL_DIMENSION).expect("matrix fixture fits u16"))
                + 1.0)
                * 0.015_625
        })
        .collect::<Vec<_>>();
    let a_buffer = create_buffer(
        &device_guard.device,
        &memory_properties,
        MATMUL_BUFFER_SIZE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let b_buffer = create_buffer(
        &device_guard.device,
        &memory_properties,
        MATMUL_BUFFER_SIZE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let output_buffer = create_buffer(
        &device_guard.device,
        &memory_properties,
        MATMUL_BUFFER_SIZE,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    write_buffer(&device_guard.device, &a_buffer, &a, MATMUL_BUFFER_SIZE)?;
    write_buffer(&device_guard.device, &b_buffer, &b, MATMUL_BUFFER_SIZE)?;

    let shader_words = spirv_words(MATMUL_SPIR_V)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_words);
    // SAFETY: The SPIR-V words were produced by GLSLC during the build.
    let shader = unsafe { device_guard.device.create_shader_module(&shader_info, None) }
        .wrap_err("failed to create the matrix-multiply shader module")?;
    let descriptor_bindings = [storage_binding(0), storage_binding(1), storage_binding(2)];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
    // SAFETY: Descriptor binding data remains alive for the call.
    let descriptor_layout = unsafe {
        device_guard
            .device
            .create_descriptor_set_layout(&descriptor_layout_info, None)
    }
    .wrap_err("failed to create the matrix-multiply descriptor layout")?;
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_layout));
    // SAFETY: The descriptor layout belongs to this live device.
    let pipeline_layout = unsafe {
        device_guard
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    }
    .wrap_err("failed to create the matrix-multiply pipeline layout")?;
    let entry_name = CString::new("main").expect("static shader entry name has no interior NUL");
    let shader_stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(&entry_name);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(shader_stage)
        .layout(pipeline_layout);
    // SAFETY: Pipeline state references live shader and layout handles.
    let pipeline = unsafe {
        device_guard.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, error)| eyre::eyre!("failed to create matrix-multiply pipeline: {error:?}"))?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no matrix-multiply pipeline"))?;

    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3);
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(std::slice::from_ref(&pool_size));
    // SAFETY: Descriptor pool data remains alive for the call.
    let descriptor_pool = unsafe {
        device_guard
            .device
            .create_descriptor_pool(&descriptor_pool_info, None)
    }
    .wrap_err("failed to create the matrix-multiply descriptor pool")?;
    let descriptor_set_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_layout));
    // SAFETY: The pool and layout belong to this live device.
    let descriptor_set = unsafe {
        device_guard
            .device
            .allocate_descriptor_sets(&descriptor_set_info)
    }
    .wrap_err("failed to allocate the matrix-multiply descriptor set")?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no matrix-multiply descriptor set"))?;
    let a_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(a_buffer.buffer)
        .offset(0)
        .range(MATMUL_BUFFER_SIZE);
    let b_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(b_buffer.buffer)
        .offset(0)
        .range(MATMUL_BUFFER_SIZE);
    let output_descriptor = vk::DescriptorBufferInfo::default()
        .buffer(output_buffer.buffer)
        .offset(0)
        .range(MATMUL_BUFFER_SIZE);
    let descriptor_writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&a_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&b_descriptor)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&output_descriptor)),
    ];
    // SAFETY: Descriptor handles and buffer infos belong to this live device.
    unsafe {
        device_guard
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }
    let query_pool = if timestamp_valid_bits > 0 {
        Some(create_timestamp_query_pool(&device_guard.device)?)
    } else {
        None
    };

    let command_pool_info =
        vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
    // SAFETY: The queue family was selected from this physical device.
    let command_pool = unsafe {
        device_guard
            .device
            .create_command_pool(&command_pool_info, None)
    }
    .wrap_err("failed to create the matrix-multiply command pool")?;
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: The command pool belongs to this live device.
    let command_buffer = unsafe {
        device_guard
            .device
            .allocate_command_buffers(&command_buffer_info)
    }
    .wrap_err("failed to allocate the matrix-multiply command buffer")?
    .into_iter()
    .next()
    .ok_or_else(|| eyre::eyre!("Vulkan returned no matrix-multiply command buffer"))?;
    let begin_info = vk::CommandBufferBeginInfo::default();
    // SAFETY: The command buffer was allocated from this live command pool.
    unsafe {
        device_guard
            .device
            .begin_command_buffer(command_buffer, &begin_info)
    }
    .wrap_err("failed to begin the matrix-multiply command buffer")?;
    if let Some(query_pool) = query_pool {
        // SAFETY: The query pool belongs to this live device and this command
        // buffer is in the recording state.
        unsafe {
            device_guard
                .device
                .cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
        }
    }
    // SAFETY: The pipeline belongs to this live device.
    unsafe {
        device_guard.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline,
        );
    }
    // SAFETY: The descriptor set layout matches the matrix shader bindings.
    unsafe {
        device_guard.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        );
    }
    if let Some(query_pool) = query_pool {
        // SAFETY: The query pool belongs to this live device and the timestamp
        // is recorded in the same command buffer as the dispatch.
        unsafe {
            device_guard.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                query_pool,
                0,
            );
        }
    }
    // SAFETY: One 16x16 workgroup covers the fixed-size matrix.
    unsafe { device_guard.device.cmd_dispatch(command_buffer, 1, 1, 1) };
    if let Some(query_pool) = query_pool {
        // SAFETY: The query pool belongs to this live device and the timestamp
        // is ordered after the dispatch in this command buffer.
        unsafe {
            device_guard.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                query_pool,
                1,
            );
        }
    }
    // SAFETY: The command buffer recording has a valid begin/dispatch sequence.
    unsafe { device_guard.device.end_command_buffer(command_buffer) }
        .wrap_err("failed to end the matrix-multiply command buffer")?;
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    // SAFETY: The fence belongs to this live device.
    let fence = unsafe {
        device_guard
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .wrap_err("failed to create the matrix-multiply fence")?;
    let started = std::time::Instant::now();
    // SAFETY: The submit info references the recorded command buffer.
    unsafe {
        device_guard
            .device
            .queue_submit(queue, std::slice::from_ref(&submit_info), fence)
    }
    .wrap_err("failed to submit the matrix-multiply command buffer")?;
    // SAFETY: The fence belongs to this live device and was signaled by the
    // submitted command buffer.
    unsafe {
        device_guard
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
    }
    .wrap_err("failed while waiting for the matrix-multiply fence")?;
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let gpu_elapsed_ns = query_pool
        .map(|query_pool| {
            read_timestamp_elapsed(&device_guard.device, query_pool, timestamp_period_ns)
        })
        .transpose()?
        .flatten();
    let output = read_buffer(
        &device_guard.device,
        &output_buffer,
        MATMUL_BUFFER_SIZE,
        MATMUL_ELEMENT_COUNT,
    )?;
    let max_abs_error = output
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let row = index / MATMUL_DIMENSION;
            let column = index % MATMUL_DIMENSION;
            let expected = (0..MATMUL_DIMENSION)
                .map(|k| a[row * MATMUL_DIMENSION + k] * b[k * MATMUL_DIMENSION + column])
                .sum::<f32>();
            (value - expected).abs()
        })
        .fold(0.0_f32, f32::max);
    let passed = max_abs_error <= 1.0e-4;

    // SAFETY: The fence is no longer needed after the wait.
    unsafe { device_guard.device.destroy_fence(fence, None) };
    // SAFETY: The command pool is no longer in use after the fence wait.
    unsafe { device_guard.device.destroy_command_pool(command_pool, None) };
    // SAFETY: Descriptor objects are no longer used by submitted work.
    unsafe {
        device_guard
            .device
            .destroy_descriptor_pool(descriptor_pool, None)
    };
    // SAFETY: The pipeline is no longer referenced.
    unsafe { device_guard.device.destroy_pipeline(pipeline, None) };
    // SAFETY: The pipeline layout is no longer referenced.
    unsafe {
        device_guard
            .device
            .destroy_pipeline_layout(pipeline_layout, None)
    };
    // SAFETY: The shader module is no longer referenced by the pipeline.
    unsafe { device_guard.device.destroy_shader_module(shader, None) };
    // SAFETY: The descriptor layout is no longer referenced.
    unsafe {
        device_guard
            .device
            .destroy_descriptor_set_layout(descriptor_layout, None)
    };
    if let Some(query_pool) = query_pool {
        // SAFETY: No command references the query pool after the fence wait.
        unsafe { device_guard.device.destroy_query_pool(query_pool, None) };
    }
    destroy_buffer(&device_guard.device, &a_buffer);
    destroy_buffer(&device_guard.device, &b_buffer);
    destroy_buffer(&device_guard.device, &output_buffer);

    Ok(VectorAddResult {
        passed,
        elapsed_ns,
        gpu_elapsed_ns,
        max_abs_error,
    })
}

struct DeviceGuard {
    device: ash::Device,
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        // SAFETY: All resources created by the probe are destroyed before the
        // device guard is dropped.
        unsafe { self.device.destroy_device(None) };
    }
}

fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn create_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    buffer_size: vk::DeviceSize,
    memory_flags: vk::MemoryPropertyFlags,
) -> Result<BufferAllocation> {
    create_buffer_with_usage(
        device,
        memory_properties,
        buffer_size,
        memory_flags,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )
}

fn create_buffer_with_usage(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    buffer_size: vk::DeviceSize,
    memory_flags: vk::MemoryPropertyFlags,
    usage: vk::BufferUsageFlags,
) -> Result<BufferAllocation> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: Buffer create-info data is valid for this call.
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .wrap_err("failed to create a vector-add storage buffer")?;
    // SAFETY: The buffer belongs to this live device.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(
        requirements.memory_type_bits,
        memory_flags,
        memory_properties,
    )
    .ok_or_else(|| eyre::eyre!("no Vulkan memory type satisfies {memory_flags:?}"))?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    // SAFETY: Allocation info references a valid memory type for this device.
    let memory = unsafe { device.allocate_memory(&allocate_info, None) }
        .wrap_err("failed to allocate vector-add storage memory")?;
    // SAFETY: The memory was allocated for this device and the buffer has no
    // previous binding.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        .wrap_err("failed to bind vector-add storage memory")?;
    Ok(BufferAllocation { buffer, memory })
}

fn upload_device_local_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    queue_family_index: u32,
    bytes: &[u8],
    operation_name: &str,
) -> Result<BufferAllocation> {
    if bytes.is_empty() {
        bail!("{operation_name} cannot be empty");
    }
    let byte_len =
        u64::try_from(bytes.len()).wrap_err_with(|| format!("{operation_name} is too large"))?;
    let staging = create_buffer_with_usage(
        device,
        memory_properties,
        byte_len,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    write_buffer_bytes(device, &staging, bytes, byte_len)?;
    let target = create_buffer_with_usage(
        device,
        memory_properties,
        byte_len,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    // SAFETY: The queue family was selected from this live physical device.
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
        .wrap_err_with(|| format!("failed to create {operation_name} upload command pool"))?;
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: The command pool belongs to this live device.
    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_info) }
        .wrap_err_with(|| format!("failed to allocate {operation_name} upload command buffer"))?
        .into_iter()
        .next()
        .ok_or_else(|| eyre::eyre!("Vulkan returned no {operation_name} upload command buffer"))?;
    // SAFETY: The command buffer belongs to the newly created command pool.
    unsafe { device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()) }
        .wrap_err_with(|| format!("failed to begin {operation_name} upload command buffer"))?;
    let copy = vk::BufferCopy::default().size(byte_len);
    // SAFETY: The staging and target buffers belong to this live device and
    // the copy range is within both allocations.
    unsafe {
        device.cmd_copy_buffer(
            command_buffer,
            staging.buffer,
            target.buffer,
            std::slice::from_ref(&copy),
        );
    };
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    // SAFETY: The barrier orders the upload before a later shader submission
    // that consumes this persistent model buffer.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    };
    // SAFETY: The command buffer contains a valid copy/barrier sequence.
    unsafe { device.end_command_buffer(command_buffer) }
        .wrap_err_with(|| format!("failed to end {operation_name} upload command buffer"))?;
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    // SAFETY: The submit info references the live command buffer and queue.
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .wrap_err_with(|| format!("failed to create {operation_name} upload fence"))?;
    // SAFETY: The submit info and fence belong to this live device.
    unsafe { device.queue_submit(queue, std::slice::from_ref(&submit_info), fence) }
        .wrap_err_with(|| format!("failed to submit {operation_name} upload"))?;
    // SAFETY: The fence belongs to this live device and is signaled by the
    // submitted upload before the staging resources are destroyed.
    unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
        .wrap_err_with(|| format!("failed while waiting for {operation_name} upload"))?;
    // SAFETY: The upload has completed and no command references these child
    // objects anymore.
    unsafe { device.destroy_fence(fence, None) };
    // SAFETY: The upload has completed and no command references this pool.
    unsafe { device.destroy_command_pool(command_pool, None) };
    destroy_buffer(device, &staging);
    Ok(target)
}

fn find_memory_type(
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Option<u32> {
    (0..memory_properties.memory_type_count).find(|index| {
        let type_bit = 1_u32.checked_shl(*index).unwrap_or(0);
        type_bits & type_bit != 0
            && memory_properties.memory_types[usize::try_from(*index).unwrap_or_default()]
                .property_flags
                .contains(required_flags)
    })
}

fn write_buffer(
    device: &ash::Device,
    allocation: &BufferAllocation,
    values: &[f32],
    buffer_size: vk::DeviceSize,
) -> Result<()> {
    write_buffer_bytes(device, allocation, weights_as_bytes(values), buffer_size)
}

fn write_buffer_bytes(
    device: &ash::Device,
    allocation: &BufferAllocation,
    bytes: &[u8],
    buffer_size: vk::DeviceSize,
) -> Result<()> {
    write_buffer_region_bytes(device, allocation, bytes, 0, buffer_size)
}

fn write_buffer_region_bytes(
    device: &ash::Device,
    allocation: &BufferAllocation,
    bytes: &[u8],
    offset: vk::DeviceSize,
    buffer_size: vk::DeviceSize,
) -> Result<()> {
    // SAFETY: The memory was allocated as host-visible and the range is within
    // the buffer allocation.
    let mapped = unsafe {
        device.map_memory(
            allocation.memory,
            offset,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )
    }
    .wrap_err("failed to map vector-add input memory")?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > buffer_size {
        bail!("Vulkan host upload exceeds the allocated buffer size");
    }
    // SAFETY: The mapped pointer covers at least bytes.len() bytes.
    unsafe { (mapped.cast::<u8>()).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
    // SAFETY: The mapped range is no longer accessed after this call.
    unsafe { device.unmap_memory(allocation.memory) };
    Ok(())
}

fn weights_as_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is a plain four-byte value and the returned byte slice has
    // exactly the same lifetime and size as the source slice.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn tokens_as_bytes(values: &[u32]) -> &[u8] {
    // SAFETY: u32 is a plain four-byte value and the returned byte slice has
    // exactly the same lifetime and size as the source slice.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn read_buffer(
    device: &ash::Device,
    allocation: &BufferAllocation,
    buffer_size: vk::DeviceSize,
    element_count: usize,
) -> Result<Vec<f32>> {
    read_buffer_region(device, allocation, buffer_size, element_count, 0)
}

fn read_buffer_region(
    device: &ash::Device,
    allocation: &BufferAllocation,
    buffer_size: vk::DeviceSize,
    element_count: usize,
    offset: vk::DeviceSize,
) -> Result<Vec<f32>> {
    // SAFETY: The memory was allocated as host-visible and the range is within
    // the buffer allocation.
    let mapped = unsafe {
        device.map_memory(
            allocation.memory,
            offset,
            buffer_size,
            vk::MemoryMapFlags::empty(),
        )
    }
    .wrap_err("failed to map vector-add output memory")?;
    // SAFETY: The mapped pointer covers the fixed-size output buffer.
    let output =
        unsafe { std::slice::from_raw_parts(mapped.cast::<f32>(), element_count).to_vec() };
    // SAFETY: The mapped range is no longer accessed after this call.
    unsafe { device.unmap_memory(allocation.memory) };
    Ok(output)
}

fn destroy_buffer(device: &ash::Device, allocation: &BufferAllocation) {
    // SAFETY: The buffer is no longer used by submitted work.
    unsafe { device.destroy_buffer(allocation.buffer, None) };
    // SAFETY: The allocation is no longer bound to a live buffer.
    unsafe { device.free_memory(allocation.memory, None) };
}

fn create_timestamp_query_pool(device: &ash::Device) -> Result<vk::QueryPool> {
    let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2);
    // SAFETY: The query-pool create info is valid and remains alive for this
    // call.
    unsafe { device.create_query_pool(&create_info, None) }
        .wrap_err("failed to create the Vulkan timestamp query pool")
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The timestamp conversion is rounded and the result is bounded to u64 nanoseconds."
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "A Vulkan timestamp delta and positive timestamp period are non-negative."
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "Timestamp conversion only needs nanosecond-scale probe timing."
)]
fn read_timestamp_elapsed(
    device: &ash::Device,
    query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
) -> Result<Option<u64>> {
    let mut timestamps = [0_u64; 2];
    // SAFETY: The query pool belongs to this live device, and the fence wait
    // guarantees both timestamp values have been written.
    unsafe {
        device.get_query_pool_results(
            query_pool,
            0,
            &mut timestamps,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .wrap_err("failed to read Vulkan timestamp queries")?;
    let Some(ticks) = timestamps[1].checked_sub(timestamps[0]) else {
        return Ok(None);
    };
    let elapsed_ns = ((ticks as f64) * f64::from(timestamp_period_ns)).round() as u64;
    Ok(Some(elapsed_ns))
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("compiled vector-add SPIR-V size is not a multiple of four bytes");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
