//! End-to-end pure-Rust `GLaDOS` inference and WAV emission.

use crate::backend::BackendKind;
use crate::backend::BackendSelection;
use crate::backend::GladosBackend;
use crate::backend::SynthesisInput;
use crate::frontend::GladosFrontend;
use crate::frontend_model::GladosPhonemizer;
use crate::model_registry::PreparedModelArtifacts;
#[cfg(feature = "vulkan")]
use crate::native_glados::Cbhg;
use crate::native_glados::GladosAcousticModel;
use crate::native_glados::GladosCpuBackend;
use crate::native_glados::GladosVocoderBackend;
use crate::native_glados::load_acoustic_model_burnpack;
use crate::native_glados::load_voice_embedding;
#[cfg(feature = "torchscript")]
use crate::native_glados::torchscript::TorchScriptRuntime;
use crate::native_glados::vocoder::HiFiGan;
use crate::native_glados::vocoder::load_hifigan_burnpack;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanBatchBufferHandle;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanContext;
#[cfg(feature = "vulkan")]
use crate::vulkan::VulkanModelBufferHandle;
#[cfg(feature = "vulkan")]
use burn::nn::BatchNorm;
#[cfg(feature = "vulkan")]
use burn::nn::Linear;
#[cfg(feature = "vulkan")]
use burn::nn::PaddingConfig1d;
#[cfg(feature = "vulkan")]
use burn::nn::conv::Conv1d;
#[cfg(feature = "vulkan")]
use burn::nn::conv::ConvTranspose1d;
#[cfg(feature = "vulkan")]
use burn::nn::gru::Gru;
#[cfg(feature = "vulkan")]
use burn::nn::lstm::Lstm;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use eyre::Context;
use eyre::bail;
use std::path::Path;
use std::time::Instant;

/// Runtime artifact roles required for local `GLaDOS` synthesis.
pub const ACOUSTIC_MODEL_ROLE: &str = "acoustic-model";
pub const VOCODER_ROLE: &str = "vocoder";
pub const FRONTEND_ROLE: &str = "frontend-dictionary";
pub const FRONTEND_MODEL_ROLE: &str = "frontend-phonemizer";
pub const VOICE_P1_ROLE: &str = "voice-p1";
pub const VOICE_P2_ROLE: &str = "voice-p2";

#[derive(Debug)]
struct BurnRuntime<AcousticBackend: Backend, VocoderBackend: Backend> {
    acoustic: GladosAcousticModel<AcousticBackend>,
    vocoder: HiFiGan<VocoderBackend>,
    voice_p1: Tensor<AcousticBackend, 2>,
    voice_p2: Tensor<AcousticBackend, 2>,
    vocoder_device: VocoderBackend::Device,
    kind: BackendKind,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanRuntime {
    context: VulkanContext,
    acoustic: GladosAcousticModel<GladosCpuBackend>,
    parity_checks: bool,
    voice_p1: Tensor<GladosCpuBackend, 2>,
    voice_p2: Tensor<GladosCpuBackend, 2>,
    pitch_proj: VulkanConv1dWeights,
    energy_proj: VulkanConv1dWeights,
    lstm_forward: VulkanLstmWeights,
    lstm_reverse: VulkanLstmWeights,
    lin: VulkanLinearWeights,
    post_proj: VulkanLinearWeights,
    postnet: VulkanPostnetWeights,
    vocoder_weights: VulkanVocoderWeights,
    reference_vocoder: Option<HiFiGan<GladosVocoderBackend>>,
    reference_vocoder_device: Option<<GladosVocoderBackend as Backend>::Device>,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanConv1dWeights {
    weights: VulkanModelBufferHandle,
    bias: VulkanModelBufferHandle,
    has_bias: bool,
    output_channels: usize,
    input_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding_left: usize,
    padding_right: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanConvTranspose1dWeights {
    weights: VulkanModelBufferHandle,
    bias: VulkanModelBufferHandle,
    has_bias: bool,
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    padding_out: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanLinearWeights {
    weights: VulkanModelBufferHandle,
    bias: VulkanModelBufferHandle,
    has_bias: bool,
    input_channels: usize,
    output_channels: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanLstmWeights {
    input_weights: VulkanModelBufferHandle,
    hidden_weights: VulkanModelBufferHandle,
    bias: VulkanModelBufferHandle,
    input_channels: usize,
    hidden_channels: usize,
    reverse: bool,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanGruWeights {
    input_weights: VulkanModelBufferHandle,
    hidden_weights: VulkanModelBufferHandle,
    input_bias: VulkanModelBufferHandle,
    hidden_bias: VulkanModelBufferHandle,
    input_channels: usize,
    hidden_channels: usize,
    reverse: bool,
    reset_after: bool,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanBatchNormWeights {
    gamma: VulkanModelBufferHandle,
    beta: VulkanModelBufferHandle,
    running_mean: VulkanModelBufferHandle,
    running_variance: VulkanModelBufferHandle,
    channels: usize,
    epsilon: f64,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanPostnetConvNormWeights {
    conv: VulkanConv1dWeights,
    norm: VulkanBatchNormWeights,
    relu: bool,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanHighwayWeights {
    transformed: VulkanLinearWeights,
    gate: VulkanLinearWeights,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanPostnetWeights {
    conv_bank: Vec<VulkanPostnetConvNormWeights>,
    conv_project1: VulkanPostnetConvNormWeights,
    conv_project2: VulkanPostnetConvNormWeights,
    pre_highway: VulkanLinearWeights,
    highways: Vec<VulkanHighwayWeights>,
    gru_forward: VulkanGruWeights,
    gru_reverse: VulkanGruWeights,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanResBlockWeights {
    convs1: Vec<VulkanConv1dWeights>,
    convs2: Vec<VulkanConv1dWeights>,
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanVocoderWeights {
    conv_pre: VulkanConv1dWeights,
    ups: Vec<VulkanConvTranspose1dWeights>,
    resblocks: Vec<Vec<VulkanResBlockWeights>>,
    conv_post: VulkanConv1dWeights,
}

#[cfg(feature = "vulkan")]
impl VulkanConv1dWeights {
    fn output_length(&self, input_length: usize) -> eyre::Result<usize> {
        let effective_kernel = self
            .dilation
            .checked_mul(self.kernel_size.saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d effective kernel overflows"))?;
        let padded_input = input_length
            .checked_add(self.padding_left)
            .and_then(|value| value.checked_add(self.padding_right))
            .ok_or_else(|| eyre::eyre!("Vulkan Conv1d padded input overflows"))?;
        if padded_input < effective_kernel {
            bail!("Vulkan Conv1d padding produces no output positions");
        }
        Ok((padded_input - effective_kernel) / self.stride + 1)
    }

    fn dispatch(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_length: usize,
    ) -> eyre::Result<usize> {
        let output_length = self.output_length(input_length)?;
        self.dispatch_to_length(batch, input, output, input_length, output_length)?;
        Ok(output_length)
    }

    fn dispatch_to_length(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_length: usize,
        output_length: usize,
    ) -> eyre::Result<()> {
        self.dispatch_to_length_with_activation(
            batch,
            input,
            output,
            input_length,
            output_length,
            false,
        )
    }

    fn dispatch_to_length_with_activation(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_length: usize,
        output_length: usize,
        relu: bool,
    ) -> eyre::Result<()> {
        let natural_output_length = self.output_length(input_length)?;
        if output_length == 0 || output_length > natural_output_length {
            bail!(
                "Vulkan Conv1d requested output length {} outside natural length {}",
                output_length,
                natural_output_length
            );
        }
        batch.dispatch_conv1d(
            self.weights,
            self.bias,
            input,
            output,
            self.input_channels,
            input_length,
            self.output_channels,
            output_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding_left,
            self.has_bias,
            relu,
        )?;
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
impl VulkanConvTranspose1dWeights {
    fn output_length(&self, input_length: usize) -> eyre::Result<usize> {
        let effective_kernel = self
            .dilation
            .checked_mul(self.kernel_size.saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d effective kernel overflows"))?;
        let untrimmed_length = input_length
            .checked_sub(1)
            .and_then(|value| value.checked_mul(self.stride))
            .and_then(|value| value.checked_add(effective_kernel))
            .and_then(|value| value.checked_add(self.padding_out))
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d output dimensions overflow"))?;
        let trim = self
            .padding
            .checked_mul(2)
            .ok_or_else(|| eyre::eyre!("Vulkan ConvTranspose1d padding overflows"))?;
        if untrimmed_length <= trim {
            bail!("Vulkan ConvTranspose1d padding removes all output positions");
        }
        Ok(untrimmed_length - trim)
    }

    fn dispatch(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        input_length: usize,
    ) -> eyre::Result<usize> {
        let output_length = self.output_length(input_length)?;
        batch.dispatch_conv_transpose1d(
            self.weights,
            self.bias,
            input,
            output,
            self.input_channels,
            input_length,
            self.output_channels,
            output_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding,
            self.padding_out,
            self.has_bias,
        )?;
        Ok(output_length)
    }
}

#[cfg(feature = "vulkan")]
impl VulkanLinearWeights {
    fn dispatch(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        sequence_length: usize,
    ) -> eyre::Result<()> {
        batch.dispatch_linear(
            self.weights,
            self.bias,
            input,
            output,
            self.input_channels,
            sequence_length,
            self.output_channels,
            self.has_bias,
        )
    }
}

#[cfg(feature = "vulkan")]
impl VulkanBatchNormWeights {
    fn dispatch(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        sequence_length: usize,
    ) -> eyre::Result<()> {
        batch.dispatch_batch_norm(
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_variance,
            input,
            output,
            self.channels,
            sequence_length,
            self.epsilon,
        )
    }
}

#[cfg(feature = "vulkan")]
impl VulkanPostnetConvNormWeights {
    fn dispatch(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        input_length: usize,
        output_length: usize,
    ) -> eyre::Result<VulkanBatchBufferHandle> {
        if self.conv.output_channels != self.norm.channels {
            bail!(
                "Vulkan postnet ConvNorm channels disagree: conv={}, norm={}",
                self.conv.output_channels,
                self.norm.channels
            );
        }
        let convolution =
            batch.alloc_buffer(tensor_elements(self.conv.output_channels, output_length)?)?;
        self.conv.dispatch_to_length_with_activation(
            batch,
            input,
            convolution,
            input_length,
            output_length,
            self.relu,
        )?;
        let normalized = batch.alloc_buffer(tensor_elements(self.norm.channels, output_length)?)?;
        self.norm
            .dispatch(batch, convolution, normalized, output_length)?;
        Ok(normalized)
    }
}

#[cfg(feature = "vulkan")]
impl VulkanGruWeights {
    fn dispatch(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        output: VulkanBatchBufferHandle,
        sequence_length: usize,
    ) -> eyre::Result<()> {
        batch.dispatch_gru(
            self.input_weights,
            self.hidden_weights,
            self.input_bias,
            self.hidden_bias,
            input,
            output,
            self.input_channels,
            self.hidden_channels,
            sequence_length,
            self.reverse,
            self.reset_after,
        )
    }
}

#[cfg(feature = "vulkan")]
impl VulkanPostnetWeights {
    #[expect(
        clippy::too_many_lines,
        reason = "The fixed CBHG graph is kept as one auditable Vulkan batch recording sequence."
    )]
    fn record(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        post_projection: &VulkanLinearWeights,
        input: VulkanBatchBufferHandle,
        mel_frames: usize,
    ) -> eyre::Result<VulkanBatchBufferHandle> {
        let mel_channels = post_projection.output_channels;
        let input_channels = self
            .conv_bank
            .first()
            .map(|branch| branch.conv.input_channels)
            .ok_or_else(|| eyre::eyre!("Vulkan postnet has no convolution bank"))?;
        if input_channels == 0 || mel_frames == 0 {
            bail!("Vulkan postnet dimensions must be non-zero");
        }
        let mel_elements = tensor_elements(input_channels, mel_frames)?;

        let bank_channels = self
            .conv_bank
            .iter()
            .map(|branch| branch.conv.output_channels)
            .sum::<usize>();
        let bank = batch.alloc_buffer(tensor_elements(bank_channels, mel_frames)?)?;
        let mut channel_offset = 0;
        for branch in &self.conv_bank {
            let branch_output = branch.dispatch(batch, input, mel_frames, mel_frames)?;
            batch.dispatch_copy_channels(
                branch_output,
                bank,
                branch.conv.output_channels,
                mel_frames,
                channel_offset,
            )?;
            channel_offset += branch.conv.output_channels;
        }
        let pooled = batch.alloc_buffer(tensor_elements(bank_channels, mel_frames)?)?;
        batch.dispatch_max_pool1d(bank, pooled, bank_channels, mel_frames, mel_frames, 2, 1, 1)?;
        let projected = self
            .conv_project1
            .dispatch(batch, pooled, mel_frames, mel_frames)?;
        let projected = self
            .conv_project2
            .dispatch(batch, projected, mel_frames, mel_frames)?;
        if self.conv_project2_output_channels() != input_channels {
            bail!(
                "Vulkan postnet residual channels {} do not match mel channels {}",
                self.conv_project2_output_channels(),
                input_channels
            );
        }
        let residual = batch.alloc_buffer(mel_elements)?;
        batch.dispatch_add(projected, input, residual, mel_elements)?;
        let highway_input = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
        self.pre_highway
            .dispatch(batch, residual, highway_input, mel_frames)?;
        let mut highway_input = highway_input;
        for highway in &self.highways {
            let gate_logits = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            highway
                .gate
                .dispatch(batch, highway_input, gate_logits, mel_frames)?;
            let gate = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            batch.dispatch_sigmoid(gate_logits, gate, tensor_elements(256, mel_frames)?)?;
            let transformed_logits = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            highway
                .transformed
                .dispatch(batch, highway_input, transformed_logits, mel_frames)?;
            let transformed = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            batch.dispatch_relu(
                transformed_logits,
                transformed,
                tensor_elements(256, mel_frames)?,
            )?;
            let gated_transformed = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            batch.dispatch_mul(
                gate,
                transformed,
                gated_transformed,
                tensor_elements(256, mel_frames)?,
            )?;
            let inverse_gate = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            batch.dispatch_one_minus(gate, inverse_gate, tensor_elements(256, mel_frames)?)?;
            let retained = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            batch.dispatch_mul(
                inverse_gate,
                highway_input,
                retained,
                tensor_elements(256, mel_frames)?,
            )?;
            let next = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
            batch.dispatch_add(
                gated_transformed,
                retained,
                next,
                tensor_elements(256, mel_frames)?,
            )?;
            highway_input = next;
        }

        let forward = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
        self.gru_forward
            .dispatch(batch, highway_input, forward, mel_frames)?;
        let reverse = batch.alloc_buffer(tensor_elements(256, mel_frames)?)?;
        self.gru_reverse
            .dispatch(batch, highway_input, reverse, mel_frames)?;
        let recurrent = batch.alloc_buffer(tensor_elements(512, mel_frames)?)?;
        batch.dispatch_copy_channels(forward, recurrent, 256, mel_frames, 0)?;
        batch.dispatch_copy_channels(reverse, recurrent, 256, mel_frames, 256)?;
        let output = batch.alloc_buffer(tensor_elements(mel_channels, mel_frames)?)?;
        post_projection.dispatch(batch, recurrent, output, mel_frames)?;
        Ok(output)
    }

    fn conv_project2_output_channels(&self) -> usize {
        self.conv_project2.norm.channels
    }
}

#[cfg(feature = "vulkan")]
type VulkanVocoderTrace = (VulkanBatchBufferHandle, usize);

#[cfg(feature = "vulkan")]
type VulkanVocoderRecord = (VulkanBatchBufferHandle, usize, Vec<VulkanVocoderTrace>);

#[cfg(feature = "vulkan")]
impl VulkanVocoderWeights {
    #[expect(
        clippy::too_many_lines,
        reason = "The fixed GLaDOS vocoder batch is kept as one auditable graph recording sequence."
    )]
    fn record(
        &self,
        batch: &mut crate::vulkan::VulkanBatch<'_>,
        input: VulkanBatchBufferHandle,
        mel_frames: usize,
        capture_trace: bool,
    ) -> eyre::Result<VulkanVocoderRecord> {
        let started = Instant::now();
        let mut captures = Vec::new();
        let mut output_length = mel_frames;
        let current = batch.alloc_buffer(tensor_elements(
            self.conv_pre.output_channels,
            self.conv_pre.output_length(output_length)?,
        )?)?;
        output_length = self
            .conv_pre
            .dispatch(batch, input, current, output_length)?;
        if capture_trace {
            captures.push((
                current,
                tensor_elements(self.conv_pre.output_channels, output_length)?,
            ));
        }
        let mut current = {
            let activated = batch.alloc_buffer(tensor_elements(
                self.conv_pre.output_channels,
                output_length,
            )?)?;
            batch.dispatch_leaky_relu(
                current,
                activated,
                tensor_elements(self.conv_pre.output_channels, output_length)?,
                0.1,
            )?;
            activated
        };
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "Vulkan vocoder input projection complete"
        );
        for (stage, (up, resblocks)) in self.ups.iter().zip(&self.resblocks).enumerate() {
            let stage_started = Instant::now();
            let upsampled_length = up.output_length(output_length)?;
            let stage_channels = up.output_channels;
            let stage_elements = tensor_elements(stage_channels, upsampled_length)?;
            let upsampled = batch.alloc_buffer(stage_elements)?;
            up.dispatch(batch, current, upsampled, output_length)?;
            let mut sum = None;
            for block in resblocks {
                if block.convs1.len() != block.convs2.len() {
                    bail!("Vulkan vocoder residual block has mismatched convolution counts");
                }
                let mut block_output = upsampled;
                for (conv1, conv2) in block.convs1.iter().zip(&block.convs2) {
                    if conv1.input_channels != stage_channels
                        || conv1.output_channels != conv2.input_channels
                        || conv2.output_channels != stage_channels
                    {
                        bail!(
                            "Vulkan vocoder residual Conv1d channel dimensions do not preserve the stage"
                        );
                    }
                    let conv1_length = conv1.output_length(upsampled_length)?;
                    if conv1_length != upsampled_length {
                        bail!("Vulkan vocoder residual first Conv1d changed sequence length");
                    }
                    let conv2_length = conv2.output_length(conv1_length)?;
                    if conv2_length != upsampled_length {
                        bail!("Vulkan vocoder residual second Conv1d changed sequence length");
                    }
                    let activated_input = batch.alloc_buffer(stage_elements)?;
                    batch.dispatch_leaky_relu(
                        block_output,
                        activated_input,
                        stage_elements,
                        0.1,
                    )?;
                    let conv1_output = batch.alloc_buffer(stage_elements)?;
                    conv1.dispatch(batch, activated_input, conv1_output, upsampled_length)?;
                    let activated_output = batch.alloc_buffer(stage_elements)?;
                    batch.dispatch_leaky_relu(
                        conv1_output,
                        activated_output,
                        stage_elements,
                        0.1,
                    )?;
                    let conv2_output = batch.alloc_buffer(stage_elements)?;
                    conv2.dispatch(batch, activated_output, conv2_output, conv1_length)?;
                    let next_block_output = batch.alloc_buffer(stage_elements)?;
                    batch.dispatch_add(
                        block_output,
                        conv2_output,
                        next_block_output,
                        stage_elements,
                    )?;
                    block_output = next_block_output;
                }
                sum = Some(if let Some(previous) = sum {
                    let next_sum = batch.alloc_buffer(stage_elements)?;
                    batch.dispatch_add(previous, block_output, next_sum, stage_elements)?;
                    next_sum
                } else {
                    block_output
                });
            }
            if resblocks.is_empty() {
                bail!("Vulkan vocoder stage has no residual blocks");
            }
            let divisor = f32::from(
                u16::try_from(resblocks.len()).expect("GLaDOS vocoder has few residual blocks"),
            );
            let sum = sum.expect("Vulkan vocoder residual sum was checked non-empty");
            let averaged = batch.alloc_buffer(stage_elements)?;
            batch.dispatch_scale(sum, averaged, stage_elements, 1.0 / divisor)?;
            current = averaged;
            output_length = upsampled_length;
            if stage + 1 != self.ups.len() {
                let activated = batch.alloc_buffer(stage_elements)?;
                batch.dispatch_leaky_relu(current, activated, stage_elements, 0.1)?;
                current = activated;
            }
            if capture_trace {
                captures.push((current, stage_elements));
            }
            tracing::info!(
                stage = stage + 1,
                elapsed_ms = stage_started.elapsed().as_millis(),
                "Vulkan vocoder stage complete"
            );
        }
        let vocoder_input_elements = tensor_elements(self.conv_post.input_channels, output_length)?;
        let waveform_elements = tensor_elements(self.conv_post.output_channels, output_length)?;
        let activated = batch.alloc_buffer(vocoder_input_elements)?;
        batch.dispatch_leaky_relu(current, activated, vocoder_input_elements, 0.01)?;
        let final_length = self.conv_post.output_length(output_length)?;
        if final_length != output_length || self.conv_post.output_channels != 1 {
            bail!("Vulkan vocoder final convolution returned an unexpected shape");
        }
        let waveform = batch.alloc_buffer(waveform_elements)?;
        self.conv_post
            .dispatch(batch, activated, waveform, output_length)?;
        let output = batch.alloc_buffer(waveform_elements)?;
        if capture_trace {
            captures.push((waveform, waveform_elements));
        }
        batch.dispatch_tanh(waveform, output, waveform_elements)?;
        if capture_trace {
            captures.push((output, waveform_elements));
        }
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "Vulkan vocoder waveform complete"
        );
        Ok((output, waveform_elements, captures))
    }
}

#[cfg(feature = "vulkan")]
fn tensor_elements(channels: usize, length: usize) -> eyre::Result<usize> {
    channels
        .checked_mul(length)
        .ok_or_else(|| eyre::eyre!("Vulkan vocoder tensor dimensions overflow"))
}

#[cfg(feature = "vulkan")]
fn dispatch_vulkan_conv1d(
    context: &VulkanContext,
    weights: &VulkanConv1dWeights,
    input: &[f32],
    input_length: usize,
) -> eyre::Result<Vec<f32>> {
    let input_elements = tensor_elements(weights.input_channels, input_length)?;
    if input.len() != input_elements {
        bail!(
            "Vulkan Conv1d input has {}, expected {}",
            input.len(),
            input_elements
        );
    }
    let output_length = weights.output_length(input_length)?;
    let output_elements = tensor_elements(weights.output_channels, output_length)?;
    let mut batch = context.begin_batch()?;
    let input_buffer = batch.alloc_buffer(input_elements)?;
    let output_buffer = batch.alloc_buffer(output_elements)?;
    batch.write_buffer(input_buffer, input)?;
    weights.dispatch(&mut batch, input_buffer, output_buffer, input_length)?;
    batch.finish(output_buffer, output_elements)
}

#[cfg(feature = "vulkan")]
fn dispatch_vulkan_lstm(
    context: &VulkanContext,
    weights: &VulkanLstmWeights,
    input: &[f32],
    sequence_length: usize,
) -> eyre::Result<Vec<f32>> {
    context.dispatch_lstm(
        weights.input_weights,
        weights.hidden_weights,
        weights.bias,
        input,
        weights.input_channels,
        weights.hidden_channels,
        sequence_length,
        weights.reverse,
    )
}

/// A loaded, native `GLaDOS` inference pipeline.
#[derive(Debug)]
pub struct GladosRuntime {
    engine: Box<dyn GladosBackend>,
    frontend: GladosFrontend,
    phonemizer: GladosPhonemizer<GladosCpuBackend>,
    sample_rate_hz: u32,
}

impl GladosRuntime {
    /// Load all native artifacts described by a prepared-model manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when a required role is missing or any Burnpack or
    /// frontend artifact cannot be loaded.
    pub fn from_prepared(
        artifacts: &PreparedModelArtifacts,
        backend_selection: BackendSelection,
    ) -> eyre::Result<Self> {
        let frontend_device = <GladosCpuBackend as Backend>::Device::default();
        let acoustic_device = <GladosCpuBackend as Backend>::Device::default();
        let frontend_path = artifacts.path_for_role(FRONTEND_ROLE)?;
        let phonemizer_path = artifacts.path_for_role(FRONTEND_MODEL_ROLE)?;

        tracing::info!("loading phonemizer model");
        let mut phonemizer = GladosPhonemizer::init(&frontend_device);
        crate::frontend_model::load_glados_phonemizer_burnpack(&mut phonemizer, phonemizer_path)?;

        tracing::info!("loading frontend and voice artifacts");
        let frontend = GladosFrontend::from_tsv(frontend_path)?;
        let engine = load_runtime_engine(artifacts, acoustic_device, backend_selection)?;
        Ok(Self {
            engine,
            frontend,
            phonemizer,
            sample_rate_hz: artifacts.manifest.sample_rate_hz,
        })
    }

    /// Generate mono floating-point audio for one English utterance.
    ///
    /// # Errors
    ///
    /// Returns an error when the voice is unknown, the input cannot be
    /// tokenized, or the generated tensor cannot be copied to the host.
    pub fn synthesize(&self, text: &str, voice: &str, alpha: f32) -> eyre::Result<Vec<f32>> {
        if !alpha.is_finite() || alpha <= 0.0 {
            bail!("alpha must be a finite positive number");
        }

        tracing::info!("phonemizing input");
        let token_values = self
            .frontend
            .tokenize_with(text, |word| self.phonemizer.phonemize_word(word))?;
        let token_count = token_values.len();
        tracing::info!(token_count, "phonemization complete");
        let input = SynthesisInput {
            tokens: &token_values,
            voice,
            alpha,
        };
        tracing::info!(backend = %self.backend_kind(), "synthesizing with backend");
        self.engine.synthesize(&input)
    }

    /// Write mono floating-point samples as a 16-bit PCM WAV file.
    ///
    /// # Errors
    ///
    /// Returns an error if the path cannot be created or the WAV sizes would
    /// overflow their 32-bit RIFF fields.
    pub fn write_wav(&self, output: &Path, samples: &[f32]) -> eyre::Result<()> {
        write_pcm16_wav(output, self.sample_rate_hz, samples)
    }

    /// Return the model sample rate.
    #[must_use]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Return the concrete backend selected for this loaded runtime.
    #[must_use]
    pub fn backend_kind(&self) -> BackendKind {
        self.engine.kind()
    }
}

fn load_runtime_engine(
    artifacts: &PreparedModelArtifacts,
    acoustic_device: <GladosCpuBackend as Backend>::Device,
    backend_selection: BackendSelection,
) -> eyre::Result<Box<dyn GladosBackend>> {
    let acoustic_path = artifacts.path_for_role(ACOUSTIC_MODEL_ROLE)?;
    let vocoder_path = artifacts.path_for_role(VOCODER_ROLE)?;
    let voice_p1_path = artifacts.path_for_role(VOICE_P1_ROLE)?;
    let voice_p2_path = artifacts.path_for_role(VOICE_P2_ROLE)?;
    let selected_backend = match backend_selection {
        BackendSelection::Auto => {
            let device_identity = crate::backend_receipts::device_identity();
            let configuration = crate::backend_receipts::BenchmarkConfiguration::default();
            let decision = crate::backend_receipts::auto_backend_decision(
                artifacts,
                &device_identity,
                &configuration,
            );
            tracing::info!(
                backend = %decision.backend,
                reason = %decision.reason,
                receipt_path = ?decision.receipt_path,
                "automatic backend selection"
            );
            BackendKind::parse(&decision.backend).unwrap_or(BackendKind::Burn)
        }
        BackendSelection::Burn => BackendKind::Burn,
        BackendSelection::BurnNdArray => BackendKind::BurnNdArray,
        BackendSelection::BurnCudaAcoustic => BackendKind::BurnCudaAcoustic,
        BackendSelection::BurnCudaFused => BackendKind::BurnCudaFused,
        BackendSelection::BurnTch => BackendKind::BurnTch,
        BackendSelection::BurnWgpu => BackendKind::BurnWgpu,
        BackendSelection::BurnVulkan => BackendKind::BurnVulkan,
        BackendSelection::LibTorch => BackendKind::LibTorch,
        BackendSelection::Vulkan => BackendKind::Vulkan,
    };

    match selected_backend {
        BackendKind::Burn => {
            load_burn_backend(acoustic_path, vocoder_path, voice_p1_path, voice_p2_path)
        }
        BackendKind::BurnNdArray => {
            load_burn_ndarray_backend(acoustic_path, vocoder_path, voice_p1_path, voice_p2_path)
        }
        BackendKind::BurnCudaAcoustic => load_burn_cuda_acoustic_backend(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
        ),
        BackendKind::BurnCudaFused => {
            load_burn_cuda_fused_backend(acoustic_path, vocoder_path, voice_p1_path, voice_p2_path)
        }
        BackendKind::BurnTch => {
            load_burn_tch_backend(acoustic_path, vocoder_path, voice_p1_path, voice_p2_path)
        }
        BackendKind::BurnWgpu => {
            load_burn_wgpu_backend(acoustic_path, vocoder_path, voice_p1_path, voice_p2_path)
        }
        BackendKind::BurnVulkan => {
            load_burn_vulkan_backend(acoustic_path, vocoder_path, voice_p1_path, voice_p2_path)
        }
        BackendKind::LibTorch => {
            let Some(model_dir) = crate::config::effective_torch_model_dir()? else {
                bail!(
                    "libtorch backend requires a TorchScript model directory; set it once with `teamy-tts config set --torch-model-dir <path>` (TEAMY_TTS_TORCH_MODEL_DIR remains an override)"
                );
            };
            load_torchscript_runtime(&model_dir, voice_p1_path, voice_p2_path)
        }
        BackendKind::Vulkan => load_vulkan_backend(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            acoustic_device,
        ),
    }
}

fn load_burn_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    Ok(Box::new(load_burn_runtime::<
        GladosCpuBackend,
        GladosVocoderBackend,
    >(
        acoustic_path,
        vocoder_path,
        voice_p1_path,
        voice_p2_path,
        &<GladosCpuBackend as Backend>::Device::default(),
        <GladosVocoderBackend as Backend>::Device::default(),
        BackendKind::Burn,
    )?))
}

fn load_burn_ndarray_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    Ok(Box::new(load_burn_runtime::<
        GladosCpuBackend,
        GladosCpuBackend,
    >(
        acoustic_path,
        vocoder_path,
        voice_p1_path,
        voice_p2_path,
        &<GladosCpuBackend as Backend>::Device::default(),
        <GladosCpuBackend as Backend>::Device::default(),
        BackendKind::BurnNdArray,
    )?))
}

fn load_burn_cuda_acoustic_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(load_burn_runtime::<
            burn::backend::Cuda,
            burn::backend::Cuda,
        >(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            &<burn::backend::Cuda as Backend>::Device::default(),
            <burn::backend::Cuda as Backend>::Device::default(),
            BackendKind::BurnCudaAcoustic,
        )?))
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (acoustic_path, vocoder_path, voice_p1_path, voice_p2_path);
        bail!("burn-cuda-acoustic is unavailable in this build; rebuild with --features cuda");
    }
}

fn load_burn_cuda_fused_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "burn-cuda-fused")]
    {
        Ok(Box::new(load_burn_runtime::<
            burn::backend::Cuda,
            burn::backend::Cuda,
        >(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            &<burn::backend::Cuda as Backend>::Device::default(),
            <burn::backend::Cuda as Backend>::Device::default(),
            BackendKind::BurnCudaFused,
        )?))
    }

    #[cfg(not(feature = "burn-cuda-fused"))]
    {
        let _ = (acoustic_path, vocoder_path, voice_p1_path, voice_p2_path);
        bail!(
            "burn-cuda-fused is unavailable in this build; rebuild with --features burn-cuda-fused"
        );
    }
}

fn load_burn_tch_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "burn-tch")]
    {
        let device_index = crate::config::effective_torch_device()?.unwrap_or(0);
        let device_index = usize::try_from(device_index)
            .wrap_err("TEAMY_TTS_TORCH_DEVICE must be a non-negative CUDA device index")?;
        let device = burn::backend::libtorch::LibTorchDevice::Cuda(device_index);
        Ok(Box::new(load_burn_runtime::<
            burn::backend::LibTorch<f32>,
            burn::backend::LibTorch<f32>,
        >(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            &device,
            device.clone(),
            BackendKind::BurnTch,
        )?))
    }

    #[cfg(not(feature = "burn-tch"))]
    {
        let _ = (acoustic_path, vocoder_path, voice_p1_path, voice_p2_path);
        bail!("burn-tch is unavailable in this build; rebuild with --features burn-tch");
    }
}

fn load_burn_wgpu_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "burn-wgpu")]
    {
        let device = <burn::backend::Wgpu as Backend>::Device::default();
        Ok(Box::new(load_burn_runtime::<
            burn::backend::Wgpu,
            burn::backend::Wgpu,
        >(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            &device,
            device.clone(),
            BackendKind::BurnWgpu,
        )?))
    }

    #[cfg(not(feature = "burn-wgpu"))]
    {
        let _ = (acoustic_path, vocoder_path, voice_p1_path, voice_p2_path);
        bail!("burn-wgpu is unavailable in this build; rebuild with --features burn-wgpu");
    }
}

fn load_burn_vulkan_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "burn-vulkan")]
    {
        let device = <burn::backend::Vulkan as Backend>::Device::default();
        let _setup = burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
            &device,
            burn::backend::wgpu::RuntimeOptions::default(),
        );
        Ok(Box::new(load_burn_runtime::<
            burn::backend::Vulkan,
            burn::backend::Vulkan,
        >(
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            &device,
            device.clone(),
            BackendKind::BurnVulkan,
        )?))
    }

    #[cfg(not(feature = "burn-vulkan"))]
    {
        let _ = (acoustic_path, vocoder_path, voice_p1_path, voice_p2_path);
        bail!("burn-vulkan is unavailable in this build; rebuild with --features burn-vulkan");
    }
}

fn load_torchscript_runtime(
    model_dir: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "torchscript")]
    {
        let glados_path = model_dir.join("glados-new.pt");
        let vocoder_gpu_path = model_dir.join("vocoder-gpu.pt");
        if !glados_path.is_file() || !vocoder_gpu_path.is_file() {
            bail!(
                "libtorch backend model directory is incomplete: {} and {} are required",
                glados_path.display(),
                vocoder_gpu_path.display()
            );
        }
        tracing::info!(model_dir = %model_dir.display(), "loading LibTorch backend");
        Ok(Box::new(TorchScriptRuntime::from_model_dir(
            model_dir,
            voice_p1_path,
            voice_p2_path,
        )?))
    }

    #[cfg(not(feature = "torchscript"))]
    {
        let _ = (model_dir, voice_p1_path, voice_p2_path);
        bail!(
            "libtorch backend is unavailable in this build; rebuild with --features all-backends (or --features torchscript for a LibTorch-only build)"
        )
    }
}

fn load_vulkan_backend(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
    acoustic_device: <GladosCpuBackend as Backend>::Device,
) -> eyre::Result<Box<dyn GladosBackend>> {
    #[cfg(feature = "vulkan")]
    {
        tracing::info!(
            "loading Vulkan backend with Burn predictor/prenet and Vulkan acoustic continuation"
        );
        let mut context = VulkanContext::new()?;
        let mut acoustic = GladosAcousticModel::init(&acoustic_device);
        load_acoustic_model_burnpack(&mut acoustic, acoustic_path)?;
        let embedding_shape = acoustic.embedding.weight.val().dims();
        let embedding_weights = acoustic
            .embedding
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy acoustic embedding weights: {error:?}"))?;
        let [embedding_vocabulary_size, embedding_dimension] = embedding_shape;
        context.prepare_embedding(
            &embedding_weights,
            embedding_vocabulary_size,
            embedding_dimension,
        )?;
        let pitch_proj = extract_vulkan_conv1d(&mut context, &acoustic.pitch_proj)?;
        let energy_proj = extract_vulkan_conv1d(&mut context, &acoustic.energy_proj)?;
        let lstm_forward = extract_vulkan_lstm(&mut context, &acoustic.lstm.forward, false)?;
        let lstm_reverse = extract_vulkan_lstm(&mut context, &acoustic.lstm.reverse, true)?;
        let lin = extract_vulkan_linear(&mut context, &acoustic.lin)?;
        let post_proj = extract_vulkan_linear(&mut context, &acoustic.post_proj)?;
        let postnet = extract_vulkan_postnet(&mut context, &acoustic.postnet)?;
        let vocoder_device = <GladosVocoderBackend as Backend>::Device::default();
        let mut vocoder = HiFiGan::<GladosVocoderBackend>::init(&vocoder_device);
        load_hifigan_burnpack(&mut vocoder, vocoder_path)?;
        let vocoder_weights = extract_vulkan_vocoder_weights(&mut context, &vocoder)?;
        let capture_vulkan_parity = matches!(
            std::env::var("TEAMY_TTS_VULKAN_PARITY").as_deref(),
            Ok("1" | "true" | "yes")
        );
        Ok(Box::new(VulkanRuntime {
            context,
            acoustic,
            parity_checks: capture_vulkan_parity,
            voice_p1: load_voice_embedding(voice_p1_path, &acoustic_device)?,
            voice_p2: load_voice_embedding(voice_p2_path, &acoustic_device)?,
            pitch_proj,
            energy_proj,
            lstm_forward,
            lstm_reverse,
            lin,
            post_proj,
            postnet,
            vocoder_weights,
            reference_vocoder: capture_vulkan_parity.then_some(vocoder),
            reference_vocoder_device: capture_vulkan_parity.then_some(vocoder_device),
        }))
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (
            acoustic_path,
            vocoder_path,
            voice_p1_path,
            voice_p2_path,
            acoustic_device,
        );
        bail!(
            "vulkan backend is unavailable in this build; rebuild with --features all-backends (or --no-default-features --features vulkan for a Vulkan-only build)"
        )
    }
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_conv1d<B: Backend>(
    context: &mut VulkanContext,
    conv: &Conv1d<B>,
) -> eyre::Result<VulkanConv1dWeights> {
    let [output_channels, group_input_channels, kernel_size] = conv.weight.val().dims();
    if conv.groups != 1 {
        bail!(
            "Vulkan Conv1d extraction only supports groups=1, got {}",
            conv.groups
        );
    }
    let (padding_left, padding_right) = match &*conv.padding {
        PaddingConfig1d::Valid => (0, 0),
        PaddingConfig1d::Explicit(value) => (*value, *value),
        PaddingConfig1d::Same => {
            bail!("Vulkan Conv1d extraction requires fixed explicit padding")
        }
    };
    let bias_values = conv
        .bias
        .as_ref()
        .map(|bias| {
            bias.val()
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| eyre::eyre!("failed to copy Vulkan Conv1d bias: {error:?}"))
        })
        .transpose()?;
    let weight_values = conv
        .weight
        .val()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to copy Vulkan Conv1d weights: {error:?}"))?;
    let weights = context.prepare_model_buffer(&weight_values)?;
    let bias = context.prepare_model_buffer(
        bias_values
            .as_deref()
            .unwrap_or(std::slice::from_ref(&0.0_f32)),
    )?;
    Ok(VulkanConv1dWeights {
        weights,
        bias,
        has_bias: bias_values.is_some(),
        output_channels,
        input_channels: group_input_channels,
        kernel_size,
        stride: conv.stride,
        dilation: conv.dilation,
        padding_left,
        padding_right,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_conv_transpose1d<B: Backend>(
    context: &mut VulkanContext,
    conv: &ConvTranspose1d<B>,
) -> eyre::Result<VulkanConvTranspose1dWeights> {
    let [input_channels, group_output_channels, kernel_size] = conv.weight.val().dims();
    if conv.groups != 1 {
        bail!(
            "Vulkan ConvTranspose1d extraction only supports groups=1, got {}",
            conv.groups
        );
    }
    let output_channels = conv.channels[1];
    if group_output_channels != output_channels {
        bail!(
            "Vulkan ConvTranspose1d weight output channels {} do not match module {}",
            group_output_channels,
            output_channels
        );
    }
    let bias_values = conv
        .bias
        .as_ref()
        .map(|bias| {
            bias.val().to_data().to_vec::<f32>().map_err(|error| {
                eyre::eyre!("failed to copy Vulkan ConvTranspose1d bias: {error:?}")
            })
        })
        .transpose()?;
    let weight_values = conv
        .weight
        .val()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to copy Vulkan ConvTranspose1d weights: {error:?}"))?;
    let weights = context.prepare_model_buffer(&weight_values)?;
    let bias = context.prepare_model_buffer(
        bias_values
            .as_deref()
            .unwrap_or(std::slice::from_ref(&0.0_f32)),
    )?;
    Ok(VulkanConvTranspose1dWeights {
        weights,
        bias,
        has_bias: bias_values.is_some(),
        input_channels,
        output_channels,
        kernel_size,
        stride: conv.stride,
        dilation: conv.dilation,
        padding: conv.padding,
        padding_out: conv.padding_out,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_linear<B: Backend>(
    context: &mut VulkanContext,
    linear: &Linear<B>,
) -> eyre::Result<VulkanLinearWeights> {
    let [input_channels, output_channels] = linear.weight.val().dims();
    let bias_values = linear
        .bias
        .as_ref()
        .map(|bias| {
            bias.val()
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| eyre::eyre!("failed to copy Vulkan linear bias: {error:?}"))
        })
        .transpose()?;
    let source_weight_values = linear
        .weight
        .val()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to copy Vulkan linear weights: {error:?}"))?;
    let mut weight_values = vec![0.0_f32; source_weight_values.len()];
    for input_channel in 0..input_channels {
        for output_channel in 0..output_channels {
            weight_values[output_channel * input_channels + input_channel] =
                source_weight_values[input_channel * output_channels + output_channel];
        }
    }
    let weights = context.prepare_model_buffer(&weight_values)?;
    let bias = context.prepare_model_buffer(
        bias_values
            .as_deref()
            .unwrap_or(std::slice::from_ref(&0.0_f32)),
    )?;
    Ok(VulkanLinearWeights {
        weights,
        bias,
        has_bias: bias_values.is_some(),
        input_channels,
        output_channels,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_lstm<B: Backend>(
    context: &mut VulkanContext,
    lstm: &Lstm<B>,
    reverse: bool,
) -> eyre::Result<VulkanLstmWeights> {
    let [input_channels, input_hidden_channels] =
        lstm.input_gate.input_transform.weight.val().dims();
    let hidden_channels = lstm.d_hidden;
    if input_hidden_channels != hidden_channels {
        bail!(
            "Vulkan LSTM input gate output size {} does not match hidden size {}",
            input_hidden_channels,
            hidden_channels
        );
    }
    let gates = [
        &lstm.input_gate,
        &lstm.forget_gate,
        &lstm.output_gate,
        &lstm.cell_gate,
    ];
    let input_weight_count = 4usize
        .checked_mul(hidden_channels)
        .and_then(|value| value.checked_mul(input_channels))
        .ok_or_else(|| eyre::eyre!("Vulkan LSTM input weights overflow"))?;
    let hidden_weight_count = 4usize
        .checked_mul(hidden_channels)
        .and_then(|value| value.checked_mul(hidden_channels))
        .ok_or_else(|| eyre::eyre!("Vulkan LSTM hidden weights overflow"))?;
    let mut input_weight_values = vec![0.0_f32; input_weight_count];
    let mut hidden_weight_values = vec![0.0_f32; hidden_weight_count];
    let mut bias_values = vec![0.0_f32; 4 * hidden_channels];
    for (gate_index, gate) in gates.into_iter().enumerate() {
        let input_weight = gate
            .input_transform
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy Vulkan LSTM input weights: {error:?}"))?;
        let hidden_weight = gate
            .hidden_transform
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy Vulkan LSTM hidden weights: {error:?}"))?;
        if input_weight.len() != input_channels * hidden_channels
            || hidden_weight.len() != hidden_channels * hidden_channels
        {
            bail!("Vulkan LSTM gate weight shape does not match the module dimensions");
        }
        for input_channel in 0..input_channels {
            for hidden_channel in 0..hidden_channels {
                let output_index = (gate_index * hidden_channels + hidden_channel) * input_channels
                    + input_channel;
                input_weight_values[output_index] =
                    input_weight[input_channel * hidden_channels + hidden_channel];
            }
        }
        for previous_hidden in 0..hidden_channels {
            for hidden_channel in 0..hidden_channels {
                let output_index = (gate_index * hidden_channels + hidden_channel)
                    * hidden_channels
                    + previous_hidden;
                hidden_weight_values[output_index] =
                    hidden_weight[previous_hidden * hidden_channels + hidden_channel];
            }
        }
        if let Some(bias) = &gate.input_transform.bias {
            let values =
                bias.val().to_data().to_vec::<f32>().map_err(|error| {
                    eyre::eyre!("failed to copy Vulkan LSTM input bias: {error:?}")
                })?;
            for hidden_channel in 0..hidden_channels {
                bias_values[gate_index * hidden_channels + hidden_channel] +=
                    values[hidden_channel];
            }
        }
        if let Some(bias) = &gate.hidden_transform.bias {
            let values = bias.val().to_data().to_vec::<f32>().map_err(|error| {
                eyre::eyre!("failed to copy Vulkan LSTM hidden bias: {error:?}")
            })?;
            for hidden_channel in 0..hidden_channels {
                bias_values[gate_index * hidden_channels + hidden_channel] +=
                    values[hidden_channel];
            }
        }
    }
    Ok(VulkanLstmWeights {
        input_weights: context.prepare_model_buffer(&input_weight_values)?,
        hidden_weights: context.prepare_model_buffer(&hidden_weight_values)?,
        bias: context.prepare_model_buffer(&bias_values)?,
        input_channels,
        hidden_channels,
        reverse,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_gru<B: Backend>(
    context: &mut VulkanContext,
    gru: &Gru<B>,
    reverse: bool,
) -> eyre::Result<VulkanGruWeights> {
    let [input_channels, hidden_channels] = gru.update_gate.input_transform.weight.val().dims();
    if hidden_channels != gru.d_hidden {
        bail!(
            "Vulkan GRU update-gate output size {} does not match hidden size {}",
            hidden_channels,
            gru.d_hidden
        );
    }
    let gates = [&gru.update_gate, &gru.reset_gate, &gru.new_gate];
    let input_weight_count = 3usize
        .checked_mul(hidden_channels)
        .and_then(|value| value.checked_mul(input_channels))
        .ok_or_else(|| eyre::eyre!("Vulkan GRU input weights overflow"))?;
    let hidden_weight_count = 3usize
        .checked_mul(hidden_channels)
        .and_then(|value| value.checked_mul(hidden_channels))
        .ok_or_else(|| eyre::eyre!("Vulkan GRU hidden weights overflow"))?;
    let mut input_weight_values = vec![0.0_f32; input_weight_count];
    let mut hidden_weight_values = vec![0.0_f32; hidden_weight_count];
    let mut input_bias_values = vec![0.0_f32; 3 * hidden_channels];
    let mut hidden_bias_values = vec![0.0_f32; 3 * hidden_channels];
    for (gate_index, gate) in gates.into_iter().enumerate() {
        let input_weight = gate
            .input_transform
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy Vulkan GRU input weights: {error:?}"))?;
        let hidden_weight = gate
            .hidden_transform
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy Vulkan GRU hidden weights: {error:?}"))?;
        if input_weight.len() != input_channels * hidden_channels
            || hidden_weight.len() != hidden_channels * hidden_channels
        {
            bail!("Vulkan GRU gate weight shape does not match the module dimensions");
        }
        for input_channel in 0..input_channels {
            for hidden_channel in 0..hidden_channels {
                let output_index = (gate_index * hidden_channels + hidden_channel) * input_channels
                    + input_channel;
                input_weight_values[output_index] =
                    input_weight[input_channel * hidden_channels + hidden_channel];
            }
        }
        for previous_hidden in 0..hidden_channels {
            for hidden_channel in 0..hidden_channels {
                let output_index = (gate_index * hidden_channels + hidden_channel)
                    * hidden_channels
                    + previous_hidden;
                hidden_weight_values[output_index] =
                    hidden_weight[previous_hidden * hidden_channels + hidden_channel];
            }
        }
        if let Some(bias) = &gate.input_transform.bias {
            let values =
                bias.val().to_data().to_vec::<f32>().map_err(|error| {
                    eyre::eyre!("failed to copy Vulkan GRU input bias: {error:?}")
                })?;
            if values.len() != hidden_channels {
                bail!("Vulkan GRU input bias shape does not match the module dimensions");
            }
            input_bias_values[gate_index * hidden_channels..(gate_index + 1) * hidden_channels]
                .copy_from_slice(&values);
        }
        if let Some(bias) = &gate.hidden_transform.bias {
            let values =
                bias.val().to_data().to_vec::<f32>().map_err(|error| {
                    eyre::eyre!("failed to copy Vulkan GRU hidden bias: {error:?}")
                })?;
            if values.len() != hidden_channels {
                bail!("Vulkan GRU hidden bias shape does not match the module dimensions");
            }
            hidden_bias_values[gate_index * hidden_channels..(gate_index + 1) * hidden_channels]
                .copy_from_slice(&values);
        }
    }
    Ok(VulkanGruWeights {
        input_weights: context.prepare_model_buffer(&input_weight_values)?,
        hidden_weights: context.prepare_model_buffer(&hidden_weight_values)?,
        input_bias: context.prepare_model_buffer(&input_bias_values)?,
        hidden_bias: context.prepare_model_buffer(&hidden_bias_values)?,
        input_channels,
        hidden_channels,
        reverse,
        reset_after: gru.reset_after,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_batch_norm<B: Backend>(
    context: &mut VulkanContext,
    batch_norm: &BatchNorm<B>,
) -> eyre::Result<VulkanBatchNormWeights> {
    let [channels] = batch_norm.gamma.val().dims();
    let gamma = batch_norm
        .gamma
        .val()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to copy Vulkan BatchNorm gamma: {error:?}"))?;
    let beta = batch_norm
        .beta
        .val()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to copy Vulkan BatchNorm beta: {error:?}"))?;
    let running_mean = batch_norm
        .running_mean
        .value()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to copy Vulkan BatchNorm running mean: {error:?}"))?;
    let running_variance = batch_norm
        .running_var
        .value()
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| {
            eyre::eyre!("failed to copy Vulkan BatchNorm running variance: {error:?}")
        })?;
    if gamma.len() != channels
        || beta.len() != channels
        || running_mean.len() != channels
        || running_variance.len() != channels
    {
        bail!("Vulkan BatchNorm statistics do not match the channel count");
    }
    Ok(VulkanBatchNormWeights {
        gamma: context.prepare_model_buffer(&gamma)?,
        beta: context.prepare_model_buffer(&beta)?,
        running_mean: context.prepare_model_buffer(&running_mean)?,
        running_variance: context.prepare_model_buffer(&running_variance)?,
        channels,
        epsilon: batch_norm.epsilon,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_postnet_conv_norm<B: Backend>(
    context: &mut VulkanContext,
    conv: &Conv1d<B>,
    batch_norm: &BatchNorm<B>,
    relu: bool,
) -> eyre::Result<VulkanPostnetConvNormWeights> {
    Ok(VulkanPostnetConvNormWeights {
        conv: extract_vulkan_conv1d(context, conv)?,
        norm: extract_vulkan_batch_norm(context, batch_norm)?,
        relu,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_postnet<B: Backend>(
    context: &mut VulkanContext,
    postnet: &Cbhg<B>,
) -> eyre::Result<VulkanPostnetWeights> {
    let conv_bank = postnet
        .conv1d_bank
        .iter()
        .map(|block| extract_vulkan_postnet_conv_norm(context, &block.conv, &block.bnorm, true))
        .collect::<eyre::Result<Vec<_>>>()?;
    let conv_project1 = extract_vulkan_postnet_conv_norm(
        context,
        &postnet.conv_project1.conv,
        &postnet.conv_project1.bnorm,
        true,
    )?;
    let conv_project2 = extract_vulkan_postnet_conv_norm(
        context,
        &postnet.conv_project2.conv,
        &postnet.conv_project2.bnorm,
        false,
    )?;
    let pre_highway = extract_vulkan_linear(context, &postnet.pre_highway)?;
    let highways = postnet
        .highways
        .iter()
        .map(|highway| {
            Ok(VulkanHighwayWeights {
                transformed: extract_vulkan_linear(context, &highway.w1)?,
                gate: extract_vulkan_linear(context, &highway.w2)?,
            })
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(VulkanPostnetWeights {
        conv_bank,
        conv_project1,
        conv_project2,
        pre_highway,
        highways,
        gru_forward: extract_vulkan_gru(context, &postnet.rnn.forward, false)?,
        gru_reverse: extract_vulkan_gru(context, &postnet.rnn.reverse, true)?,
    })
}

#[cfg(feature = "vulkan")]
fn extract_vulkan_vocoder_weights<B: Backend>(
    context: &mut VulkanContext,
    vocoder: &HiFiGan<B>,
) -> eyre::Result<VulkanVocoderWeights> {
    let ups = vocoder
        .ups
        .iter()
        .map(|conv| extract_vulkan_conv_transpose1d(context, conv))
        .collect::<eyre::Result<Vec<_>>>()?;
    let resblocks = vocoder
        .resblocks
        .iter()
        .map(|stage| {
            stage
                .iter()
                .map(|block| {
                    Ok(VulkanResBlockWeights {
                        convs1: block
                            .convs1
                            .iter()
                            .map(|conv| extract_vulkan_conv1d(context, conv))
                            .collect::<eyre::Result<Vec<_>>>()?,
                        convs2: block
                            .convs2
                            .iter()
                            .map(|conv| extract_vulkan_conv1d(context, conv))
                            .collect::<eyre::Result<Vec<_>>>()?,
                    })
                })
                .collect::<eyre::Result<Vec<_>>>()
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(VulkanVocoderWeights {
        conv_pre: extract_vulkan_conv1d(context, &vocoder.conv_pre)?,
        ups,
        resblocks,
        conv_post: extract_vulkan_conv1d(context, &vocoder.conv_post)?,
    })
}

fn load_burn_runtime<AcousticBackend: Backend, VocoderBackend: Backend>(
    acoustic_path: &Path,
    vocoder_path: &Path,
    voice_p1_path: &Path,
    voice_p2_path: &Path,
    acoustic_device: &AcousticBackend::Device,
    vocoder_device: VocoderBackend::Device,
    kind: BackendKind,
) -> eyre::Result<BurnRuntime<AcousticBackend, VocoderBackend>> {
    tracing::info!("loading acoustic model");
    let mut acoustic = GladosAcousticModel::init(acoustic_device);
    load_acoustic_model_burnpack(&mut acoustic, acoustic_path)?;
    tracing::info!("loading vocoder model");
    let mut vocoder = HiFiGan::init(&vocoder_device);
    load_hifigan_burnpack(&mut vocoder, vocoder_path)?;
    Ok(BurnRuntime {
        acoustic,
        vocoder,
        voice_p1: load_voice_embedding(voice_p1_path, acoustic_device)?,
        voice_p2: load_voice_embedding(voice_p2_path, acoustic_device)?,
        vocoder_device,
        kind,
    })
}

impl<AcousticBackend: Backend, VocoderBackend: Backend> GladosBackend
    for BurnRuntime<AcousticBackend, VocoderBackend>
where
    AcousticBackend::IntElem: From<i32>,
{
    fn kind(&self) -> BackendKind {
        self.kind
    }

    fn synthesize(&self, input: &SynthesisInput<'_>) -> eyre::Result<Vec<f32>> {
        let speaker = match input.voice {
            "p1" => self.voice_p1.clone(),
            "p2" => self.voice_p2.clone(),
            other => bail!("unknown voice {other:?}; expected p1 or p2"),
        };
        let device = speaker.device();
        let token_count = input.tokens.len();
        let token_values = input
            .tokens
            .iter()
            .copied()
            .map(AcousticBackend::IntElem::from)
            .collect();
        let tokens = Tensor::<AcousticBackend, 2, Int>::from_data(
            TensorData::new(token_values, [1, token_count]),
            &device,
        );
        tracing::info!("generating acoustic mel spectrogram");
        let acoustic_started = Instant::now();
        let mel = self
            .acoustic
            .generate(tokens, speaker, input.alpha)
            .mel_post;
        tracing::info!(
            elapsed_ms = acoustic_started.elapsed().as_millis(),
            "acoustic mel spectrogram complete"
        );
        tracing::info!("generating waveform with vocoder");
        let [batch_size, mel_channels, mel_frames] = mel.dims();
        let mel_values = mel.to_data().to_vec::<f32>().map_err(|error| {
            eyre::eyre!("failed to copy acoustic mel spectrogram to the vocoder: {error:?}")
        })?;
        let vocoder_mel = Tensor::<VocoderBackend, 3>::from_data(
            TensorData::new(mel_values, [batch_size, mel_channels, mel_frames]),
            &self.vocoder_device,
        );
        let audio = self.vocoder.forward(vocoder_mel);
        audio
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy generated audio to the host: {error:?}"))
    }
}

#[cfg(feature = "vulkan")]
impl GladosBackend for VulkanRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The Vulkan backend keeps embedding, acoustic, vocoder, and parity gates visible."
    )]
    fn synthesize(&self, input: &SynthesisInput<'_>) -> eyre::Result<Vec<f32>> {
        let speaker = match input.voice {
            "p1" => self.voice_p1.clone(),
            "p2" => self.voice_p2.clone(),
            other => bail!("unknown voice {other:?}; expected p1 or p2"),
        };
        let device = speaker.device();
        let token_count = input.tokens.len();
        let tokens = Tensor::<GladosCpuBackend, 2, Int>::from_data(
            TensorData::new(input.tokens.to_vec(), [1, token_count]),
            &device,
        );
        let vulkan_tokens = input
            .tokens
            .iter()
            .copied()
            .map(|token| {
                u32::try_from(token)
                    .map_err(|error| eyre::eyre!("token ID {token} cannot fit Vulkan: {error}"))
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        let vulkan_embedding = self.context.dispatch_prepared_embedding(&vulkan_tokens)?;
        let embedded = self.acoustic.embedding.forward(tokens.clone());
        let cpu_embedding = embedded
            .clone()
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| {
                eyre::eyre!("failed to copy CPU embedding for Vulkan parity: {error:?}")
            })?;
        let max_abs_error = vulkan_embedding
            .iter()
            .zip(cpu_embedding.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        if max_abs_error > 1.0e-4 {
            bail!(
                "Vulkan embedding parity check failed: max absolute error {max_abs_error} exceeds 0.0001"
            );
        }
        tracing::debug!(max_abs_error, "Vulkan embedding parity check passed");

        let reference_output = self.parity_checks.then(|| {
            tracing::info!("generating Burn reference graph for Vulkan parity");
            let acoustic_started = Instant::now();
            let output = self
                .acoustic
                .generate(tokens.clone(), speaker.clone(), input.alpha);
            tracing::info!(
                elapsed_ms = acoustic_started.elapsed().as_millis(),
                "Burn reference graph complete"
            );
            output
        });
        tracing::info!("generating acoustic predictor and prenet prefix");
        let pitch_condition_scores =
            self.acoustic
                .pitch_cond_pred
                .forward(tokens.clone(), speaker.clone(), 1.0);
        let pitch_conditions = pitch_condition_scores
            .squeeze_dim::<2>(0)
            .argmax(1)
            .squeeze_dim::<1>(1)
            .unsqueeze_dim::<2>(0);
        let durations = self
            .acoustic
            .dur_pred
            .forward(
                tokens.clone(),
                pitch_conditions.clone(),
                speaker.clone(),
                input.alpha,
            )
            .squeeze_dim::<2>(2);
        let duration_values = durations
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy acoustic durations: {error:?}"))?;
        let durations = if duration_values.iter().sum::<f32>() <= 0.0 {
            Tensor::full(durations.dims(), 2.0, &durations.device())
        } else {
            durations
        };
        let pitch = self
            .acoustic
            .pitch_pred
            .forward(
                tokens.clone(),
                pitch_conditions.clone(),
                speaker.clone(),
                1.0,
            )
            .transpose();
        let energy = self
            .acoustic
            .energy_pred
            .forward(tokens.clone(), speaker.clone(), 1.0)
            .transpose();
        let prenet = self.acoustic.prenet.forward(embedded.transpose());
        let [_, sequence_length, _] = prenet.dims();
        let speaker_sequence = speaker.unsqueeze_dim(1).repeat(&[1, sequence_length, 1]);
        let base_conditioning = Tensor::cat(vec![prenet, speaker_sequence], 2);
        let [pitch_batch_size, pitch_input_channels, pitch_length] = pitch.dims();
        let [energy_batch_size, energy_input_channels, energy_length] = energy.dims();
        if pitch_batch_size != 1
            || energy_batch_size != 1
            || pitch_input_channels != self.pitch_proj.input_channels
            || energy_input_channels != self.energy_proj.input_channels
            || pitch_length != energy_length
        {
            bail!(
                "Vulkan acoustic condition inputs have unsupported shapes: pitch={:?}, energy={:?}",
                pitch.dims(),
                energy.dims()
            );
        }
        let pitch_values = pitch
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy acoustic pitch to Vulkan: {error:?}"))?;
        let energy_values = energy
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy acoustic energy to Vulkan: {error:?}"))?;
        let vulkan_pitch_projection =
            dispatch_vulkan_conv1d(&self.context, &self.pitch_proj, &pitch_values, pitch_length)?;
        let vulkan_energy_projection = dispatch_vulkan_conv1d(
            &self.context,
            &self.energy_proj,
            &energy_values,
            energy_length,
        )?;
        if let Some(reference_output) = reference_output.as_ref() {
            let reference_pitch_projection = reference_output
                .pitch_projection
                .clone()
                .transpose()
                .squeeze_dim::<2>(0)
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| eyre::eyre!("failed to copy Burn pitch projection: {error:?}"))?;
            let reference_energy_projection = reference_output
                .energy_projection
                .clone()
                .transpose()
                .squeeze_dim::<2>(0)
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| eyre::eyre!("failed to copy Burn energy projection: {error:?}"))?;
            let (pitch_projection_max_abs_error, pitch_projection_relative_rms_error) =
                compare_vulkan_trace(&reference_pitch_projection, &vulkan_pitch_projection)?;
            let (energy_projection_max_abs_error, energy_projection_relative_rms_error) =
                compare_vulkan_trace(&reference_energy_projection, &vulkan_energy_projection)?;
            if pitch_projection_max_abs_error > 1.0e-3 {
                bail!(
                    "Vulkan pitch projection parity check failed: max absolute error {pitch_projection_max_abs_error} exceeds 0.001"
                );
            }
            if energy_projection_max_abs_error > 1.0e-3 {
                bail!(
                    "Vulkan energy projection parity check failed: max absolute error {energy_projection_max_abs_error} exceeds 0.001"
                );
            }
            tracing::debug!(
                max_abs_error = pitch_projection_max_abs_error,
                relative_rms_error = pitch_projection_relative_rms_error,
                "Vulkan acoustic pitch projection parity check passed"
            );
            tracing::debug!(
                max_abs_error = energy_projection_max_abs_error,
                relative_rms_error = energy_projection_relative_rms_error,
                "Vulkan acoustic energy projection parity check passed"
            );
        }
        tracing::info!("Vulkan acoustic condition projections complete");
        let base_conditioning_values = base_conditioning
            .clone()
            .transpose()
            .squeeze_dim::<2>(0)
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("failed to copy acoustic base conditioning: {error:?}"))?;
        if base_conditioning_values.len() != vulkan_pitch_projection.len()
            || base_conditioning_values.len() != vulkan_energy_projection.len()
        {
            bail!(
                "Vulkan acoustic condition projection length mismatch: base={}, pitch={}, energy={}",
                base_conditioning_values.len(),
                vulkan_pitch_projection.len(),
                vulkan_energy_projection.len()
            );
        }
        let conditioning_values = base_conditioning_values
            .into_iter()
            .zip(vulkan_pitch_projection)
            .zip(vulkan_energy_projection)
            .map(|((base, pitch), energy)| base + pitch + energy)
            .collect::<Vec<_>>();
        let [
            conditioning_batch_size,
            conditioning_token_count,
            conditioning_channels,
        ] = base_conditioning.dims();
        if conditioning_batch_size != 1 || conditioning_token_count != pitch_length {
            bail!(
                "Vulkan acoustic conditioning has unsupported shape: {:?}",
                base_conditioning.dims()
            );
        }
        let duration_values = durations.to_data().to_vec::<f32>().map_err(|error| {
            eyre::eyre!("failed to copy acoustic durations to Vulkan: {error:?}")
        })?;
        let regulated_frame_count = duration_values
            .iter()
            .map(|duration| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "Durations are clamped non-negative and the model's frame count is usize-sized."
                )]
                let repeat_count = (duration + 0.5).max(0.0) as usize;
                repeat_count
            })
            .sum::<usize>()
            .max(1);
        let vulkan_regulated = self.context.dispatch_length_regulate(
            &conditioning_values,
            conditioning_channels,
            conditioning_token_count,
            &duration_values,
            regulated_frame_count,
        )?;
        if let Some(reference_output) = reference_output.as_ref() {
            let reference_regulated_values = reference_output
                .regulated
                .clone()
                .transpose()
                .squeeze_dim::<2>(0)
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| {
                    eyre::eyre!("failed to copy Burn regulated conditioning: {error:?}")
                })?;
            let (length_regulation_max_abs_error, length_regulation_relative_rms_error) =
                compare_vulkan_trace(&reference_regulated_values, &vulkan_regulated)?;
            if length_regulation_max_abs_error > 1.0e-4 {
                bail!(
                    "Vulkan length-regulation parity check failed: max absolute error {length_regulation_max_abs_error} exceeds 0.0001"
                );
            }
            tracing::debug!(
                max_abs_error = length_regulation_max_abs_error,
                relative_rms_error = length_regulation_relative_rms_error,
                "Vulkan acoustic length-regulation parity check passed"
            );
        }
        tracing::info!("Vulkan acoustic length regulation complete");
        if self.lstm_forward.input_channels != conditioning_channels
            || self.lstm_reverse.input_channels != conditioning_channels
            || self.lstm_forward.hidden_channels != self.lstm_reverse.hidden_channels
        {
            bail!(
                "Vulkan acoustic LSTM shape does not match regulated conditioning: forward={}x{}, reverse={}x{}, conditioning_channels={}",
                self.lstm_forward.input_channels,
                self.lstm_forward.hidden_channels,
                self.lstm_reverse.input_channels,
                self.lstm_reverse.hidden_channels,
                conditioning_channels
            );
        }
        let mut regulated_sequence_values = vec![0.0_f32; vulkan_regulated.len()];
        for channel in 0..conditioning_channels {
            for frame in 0..regulated_frame_count {
                regulated_sequence_values[frame * conditioning_channels + channel] =
                    vulkan_regulated[channel * regulated_frame_count + frame];
            }
        }
        let vulkan_lstm_forward = dispatch_vulkan_lstm(
            &self.context,
            &self.lstm_forward,
            &regulated_sequence_values,
            regulated_frame_count,
        )?;
        let vulkan_lstm_reverse = dispatch_vulkan_lstm(
            &self.context,
            &self.lstm_reverse,
            &regulated_sequence_values,
            regulated_frame_count,
        )?;
        let mut vulkan_lstm =
            Vec::with_capacity(vulkan_lstm_forward.len() + vulkan_lstm_reverse.len());
        for frame in 0..regulated_frame_count {
            let offset = frame * self.lstm_forward.hidden_channels;
            vulkan_lstm.extend_from_slice(
                &vulkan_lstm_forward[offset..offset + self.lstm_forward.hidden_channels],
            );
            vulkan_lstm.extend_from_slice(
                &vulkan_lstm_reverse[offset..offset + self.lstm_reverse.hidden_channels],
            );
        }
        if let Some(reference_output) = reference_output.as_ref() {
            let reference_lstm_values = reference_output
                .lstm_output
                .clone()
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| {
                    eyre::eyre!("failed to copy Burn acoustic LSTM output: {error:?}")
                })?;
            let (lstm_max_abs_error, lstm_relative_rms_error) =
                compare_vulkan_trace(&reference_lstm_values, &vulkan_lstm)?;
            if lstm_max_abs_error > 1.0e-3 {
                bail!(
                    "Vulkan acoustic LSTM parity check failed: max absolute error {lstm_max_abs_error} exceeds 0.001"
                );
            }
            tracing::debug!(
                max_abs_error = lstm_max_abs_error,
                relative_rms_error = lstm_relative_rms_error,
                "Vulkan acoustic LSTM parity check passed"
            );
        }
        tracing::info!("Vulkan acoustic LSTM complete");
        let mut lstm_channel_major = vec![0.0_f32; vulkan_lstm.len()];
        let lstm_output_channels = self.lin.input_channels;
        if lstm_output_channels
            != self.lstm_forward.hidden_channels + self.lstm_reverse.hidden_channels
        {
            bail!(
                "Vulkan acoustic mel projection expects {} LSTM channels, got {}",
                self.lin.input_channels,
                self.lstm_forward.hidden_channels + self.lstm_reverse.hidden_channels
            );
        }
        for frame in 0..regulated_frame_count {
            for channel in 0..lstm_output_channels {
                lstm_channel_major[channel * regulated_frame_count + frame] =
                    vulkan_lstm[frame * lstm_output_channels + channel];
            }
        }
        let vulkan_mel_values = self.context.dispatch_linear(
            self.lin.weights,
            self.lin.bias,
            &lstm_channel_major,
            self.lin.input_channels,
            regulated_frame_count,
            self.lin.output_channels,
            self.lin.has_bias,
        )?;
        if let Some(reference_output) = reference_output.as_ref() {
            let reference_mel_values = reference_output
                .mel
                .clone()
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| eyre::eyre!("failed to copy Burn acoustic mel: {error:?}"))?;
            let (mel_projection_max_abs_error, mel_projection_relative_rms_error) =
                compare_vulkan_trace(&reference_mel_values, &vulkan_mel_values)?;
            if mel_projection_max_abs_error > 1.0e-3 {
                bail!(
                    "Vulkan acoustic mel projection parity check failed: max absolute error {mel_projection_max_abs_error} exceeds 0.001"
                );
            }
            tracing::debug!(
                max_abs_error = mel_projection_max_abs_error,
                relative_rms_error = mel_projection_relative_rms_error,
                "Vulkan acoustic mel projection parity check passed"
            );
        }
        tracing::info!("Vulkan acoustic mel projection complete");
        tracing::info!("generating postnet and waveform with Vulkan kernels");
        let mel_channels = self.post_proj.output_channels;
        let mel_frames = regulated_frame_count;
        if mel_channels != self.vocoder_weights.conv_pre.input_channels {
            bail!(
                "Vulkan vocoder expects {} mel channels, got {}",
                self.vocoder_weights.conv_pre.input_channels,
                mel_channels
            );
        }
        let capture_trace = self.reference_vocoder.is_some();
        let mut batch = self.context.begin_batch()?;
        let mel_input = batch.alloc_buffer(vulkan_mel_values.len())?;
        batch.write_buffer(mel_input, &vulkan_mel_values)?;
        let postnet_output =
            self.postnet
                .record(&mut batch, &self.post_proj, mel_input, mel_frames)?;
        let (audio_handle, audio_elements, trace_handles) =
            self.vocoder_weights
                .record(&mut batch, postnet_output, mel_frames, capture_trace)?;
        let mut outputs = vec![
            (postnet_output, tensor_elements(mel_channels, mel_frames)?),
            (audio_handle, audio_elements),
        ];
        if capture_trace {
            outputs.extend(trace_handles.iter().copied());
        }
        let mut readbacks = batch.finish_outputs(&outputs)?;
        let mel_values = readbacks.remove(0);
        let audio = readbacks.remove(0);
        let candidate_traces = capture_trace.then_some(readbacks);
        if let Some(reference_output) = reference_output.as_ref() {
            let reference_mel_values = reference_output
                .mel_post
                .clone()
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| {
                    eyre::eyre!("failed to copy acoustic mel spectrogram: {error:?}")
                })?;
            let (post_projection_max_abs_error, post_projection_relative_rms_error) =
                compare_vulkan_trace(&reference_mel_values, &mel_values)?;
            if post_projection_max_abs_error > 1.0e-3 {
                bail!(
                    "Vulkan acoustic post projection parity check failed: max absolute error {post_projection_max_abs_error} exceeds 0.001"
                );
            }
            tracing::debug!(
                max_abs_error = post_projection_max_abs_error,
                relative_rms_error = post_projection_relative_rms_error,
                "Vulkan acoustic post projection parity check passed"
            );
        }
        tracing::info!("Vulkan acoustic post projection complete");
        if let (Some(reference_vocoder), Some(reference_device), Some(candidate_traces)) = (
            self.reference_vocoder.as_ref(),
            self.reference_vocoder_device.as_ref(),
            candidate_traces,
        ) {
            let reference_mel = Tensor::<GladosVocoderBackend, 3>::from_data(
                TensorData::new(mel_values.clone(), [1, mel_channels, mel_frames]),
                reference_device,
            );
            let reference_traces = reference_vocoder.trace_forward(reference_mel);
            if reference_traces.len() != candidate_traces.len() {
                bail!(
                    "Vulkan parity trace count differs: reference={}, candidate={}",
                    reference_traces.len(),
                    candidate_traces.len()
                );
            }
            for (stage, (reference, candidate)) in
                reference_traces.iter().zip(candidate_traces).enumerate()
            {
                let reference = reference.to_data().to_vec::<f32>().map_err(|error| {
                    eyre::eyre!("failed to copy Burn Vulkan parity trace {stage}: {error:?}")
                })?;
                let (max_abs_error, relative_rms_error) =
                    compare_vulkan_trace(&reference, &candidate)?;
                tracing::debug!(
                    stage,
                    max_abs_error,
                    relative_rms_error,
                    "Vulkan parity trace"
                );
            }
        }
        Ok(audio)
    }
}

#[cfg(feature = "vulkan")]
fn compare_vulkan_trace(reference: &[f32], candidate: &[f32]) -> eyre::Result<(f64, f64)> {
    if reference.len() != candidate.len() {
        bail!(
            "Vulkan parity trace shape differs: reference={}, candidate={}",
            reference.len(),
            candidate.len()
        );
    }
    let mut reference_energy = 0.0_f64;
    let mut difference_energy = 0.0_f64;
    let mut max_abs_error = 0.0_f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        if !reference.is_finite() || !candidate.is_finite() {
            bail!("Vulkan parity trace contains non-finite values");
        }
        let reference = f64::from(reference);
        let candidate = f64::from(candidate);
        reference_energy += reference * reference;
        let difference = reference - candidate;
        difference_energy += difference * difference;
        max_abs_error = max_abs_error.max(difference.abs());
    }
    let relative_rms_error = if reference_energy <= f64::EPSILON {
        if difference_energy <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (difference_energy / reference_energy).sqrt()
    };
    Ok((max_abs_error, relative_rms_error))
}

#[cfg(feature = "torchscript")]
impl GladosBackend for TorchScriptRuntime {
    fn kind(&self) -> BackendKind {
        BackendKind::LibTorch
    }

    fn synthesize(&self, input: &SynthesisInput<'_>) -> eyre::Result<Vec<f32>> {
        tracing::info!("generating waveform with LibTorch");
        let started = Instant::now();
        let values = self.synthesize(input.tokens, input.voice, input.alpha)?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "LibTorch synthesis complete"
        );
        Ok(values)
    }
}

fn write_pcm16_wav(output: &Path, sample_rate_hz: u32, samples: &[f32]) -> eyre::Result<()> {
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| eyre::eyre!("WAV data size overflows usize"))?;
    let riff_size = 36usize
        .checked_add(data_bytes)
        .ok_or_else(|| eyre::eyre!("WAV RIFF size overflows usize"))?;
    let riff_size =
        u32::try_from(riff_size).map_err(|error| eyre::eyre!("WAV is too large: {error}"))?;
    let data_bytes =
        u32::try_from(data_bytes).map_err(|error| eyre::eyre!("WAV is too large: {error}"))?;
    let byte_rate = sample_rate_hz
        .checked_mul(2)
        .ok_or_else(|| eyre::eyre!("WAV byte rate overflows u32"))?;

    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate_hz.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        let sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The sample was clamped to the signed 16-bit audio range before conversion."
        )]
        let integer = (sample * 32_767.0).round() as i16;
        bytes.extend_from_slice(&integer.to_le_bytes());
    }

    std::fs::write(output, bytes)
        .wrap_err_with(|| format!("failed to write WAV output {}", output.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_pcm16_wav_header() {
        let path = std::env::temp_dir().join(format!("teamy-tts-test-{}.wav", std::process::id()));
        write_pcm16_wav(&path, 22_050, &[0.0, 1.0, -1.0]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 6);
        std::fs::remove_file(path).unwrap();
    }
}
