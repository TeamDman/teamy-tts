#![expect(
    clippy::disallowed_types,
    reason = "Burn's Module derive uses serde internally for model records."
)]

use burn::module::Module;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::conv::ConvTranspose1d;
use burn::nn::conv::ConvTranspose1dConfig;
use burn::tensor::Tensor;
use burn::tensor::activation::leaky_relu;
use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use burn_store::PytorchStore;
use eyre::WrapErr;
use eyre::bail;
use std::path::Path;
use std::time::Instant;

const CHANNELS: [usize; 5] = [512, 256, 128, 64, 32];
const UPSAMPLE_STRIDES: [usize; 4] = [8, 8, 2, 2];
const UPSAMPLE_KERNELS: [usize; 4] = [16, 16, 4, 4];
const UPSAMPLE_PADDING: [usize; 4] = [4, 4, 1, 1];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];

#[derive(Module, Debug)]
pub struct HiFiGanResBlock<B: Backend> {
    pub convs1: Vec<Conv1d<B>>,
    pub convs2: Vec<Conv1d<B>>,
}

impl<B: Backend> HiFiGanResBlock<B> {
    fn init(channels: usize, kernel_size: usize, device: &B::Device) -> Self {
        let convs1 = RESBLOCK_DILATIONS
            .into_iter()
            .map(|dilation| {
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_dilation(dilation)
                    .with_padding(PaddingConfig1d::Explicit((kernel_size - 1) * dilation / 2))
                    .init(device)
            })
            .collect();
        let convs2 = (0..RESBLOCK_DILATIONS.len())
            .map(|_| {
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_padding(PaddingConfig1d::Explicit(kernel_size / 2))
                    .init(device)
            })
            .collect();
        Self { convs1, convs2 }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = input;
        for (conv1, conv2) in self.convs1.iter().zip(&self.convs2) {
            let mut residual = leaky_relu(output.clone(), 0.1);
            residual = leaky_relu(conv1.forward(residual), 0.1);
            residual = conv2.forward(residual);
            output = output + residual;
        }
        output
    }
}

/// Pure-Rust HiFi-GAN-compatible vocoder matching the upstream GPU graph.
#[derive(Module, Debug)]
pub struct HiFiGan<B: Backend> {
    pub conv_pre: Conv1d<B>,
    pub ups: Vec<ConvTranspose1d<B>>,
    pub resblocks: Vec<Vec<HiFiGanResBlock<B>>>,
    pub conv_post: Conv1d<B>,
}

impl<B: Backend> HiFiGan<B> {
    /// Initialize the upstream four-stage `GLaDOS` vocoder architecture.
    #[must_use]
    pub fn init(device: &B::Device) -> Self {
        let conv_pre = Conv1dConfig::new(80, CHANNELS[0], 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);
        let mut ups = Vec::new();
        let mut resblocks = Vec::new();
        for stage in 0..UPSAMPLE_STRIDES.len() {
            ups.push(
                ConvTranspose1dConfig::new(
                    [CHANNELS[stage], CHANNELS[stage + 1]],
                    UPSAMPLE_KERNELS[stage],
                )
                .with_stride(UPSAMPLE_STRIDES[stage])
                .with_padding(UPSAMPLE_PADDING[stage])
                .init(device),
            );
            resblocks.push(
                (0..RESBLOCK_KERNELS.len())
                    .map(|block| {
                        HiFiGanResBlock::init(CHANNELS[stage + 1], RESBLOCK_KERNELS[block], device)
                    })
                    .collect(),
            );
        }
        let conv_post = Conv1dConfig::new(32, 1, 7)
            .with_padding(PaddingConfig1d::Explicit(3))
            .init(device);
        Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
        }
    }

    /// Convert a mel spectrogram `[batch, 80, frames]` into audio samples.
    ///
    /// # Panics
    ///
    /// Panics if the fixed architecture has a stage without its three
    /// residual blocks.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let started = Instant::now();
        let output = self.conv_pre.forward(mel);
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "vocoder input projection complete"
        );
        self.forward_from_conv_pre(output, started)
    }

    /// Return the raw input projection, each completed upsampling stage, and
    /// the final waveform for backend parity diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the fixed architecture has a stage without its three
    /// residual blocks.
    pub fn trace_forward(&self, mel: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
        let mut traces = Vec::with_capacity(self.ups.len() + 2);
        let mut output = self.conv_pre.forward(mel);
        traces.push(output.clone());
        output = leaky_relu(output, 0.1);
        for (stage, (up, resblocks)) in self.ups.iter().zip(&self.resblocks).enumerate() {
            output = up.forward(output);
            let mut sum = None;
            for resblock in resblocks {
                let value = resblock.forward(output.clone());
                sum = Some(match sum {
                    Some(sum) => sum + value,
                    None => value,
                });
            }
            output = sum
                .expect("each HiFi-GAN stage should have residual blocks")
                .div_scalar(3.0);
            if stage + 1 != self.ups.len() {
                output = leaky_relu(output, 0.1);
            }
            traces.push(output.clone());
        }
        output = leaky_relu(output, 0.01);
        output = self.conv_post.forward(output);
        traces.push(output.clone());
        traces.push(output.tanh());
        traces
    }

    /// Finish HiFi-GAN synthesis from the output of `conv_pre`.
    ///
    /// This boundary lets an accelerated candidate replace the input
    /// projection while retaining the exact native reference graph for the
    /// remaining stages during incremental backend development.
    ///
    /// # Panics
    ///
    /// Panics if the fixed architecture has a stage without its three
    /// residual blocks.
    pub fn forward_from_conv_pre(&self, output: Tensor<B, 3>, started: Instant) -> Tensor<B, 3> {
        let mut output = leaky_relu(output, 0.1);
        for (stage, (up, resblocks)) in self.ups.iter().zip(&self.resblocks).enumerate() {
            let stage_started = Instant::now();
            output = up.forward(output);
            let mut sum = None;
            for resblock in resblocks {
                let value = resblock.forward(output.clone());
                sum = Some(match sum {
                    Some(sum) => sum + value,
                    None => value,
                });
            }
            output = sum
                .expect("each HiFi-GAN stage should have residual blocks")
                .div_scalar(3.0);
            if stage + 1 != self.ups.len() {
                output = leaky_relu(output, 0.1);
            }
            tracing::info!(
                stage = stage + 1,
                elapsed_ms = stage_started.elapsed().as_millis(),
                "vocoder stage complete"
            );
        }
        output = leaky_relu(output, 0.01);
        let output = self.conv_post.forward(output).tanh();
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "vocoder waveform complete"
        );
        output
    }
}

/// Load the converter-produced vocoder state dictionary.
///
/// # Errors
///
/// Returns an error if the state dictionary cannot be read or does not fully
/// populate the supplied vocoder.
pub fn load_hifigan_pytorch<B: Backend>(module: &mut HiFiGan<B>, path: &Path) -> eyre::Result<()> {
    let mut store = PytorchStore::from_file(path).allow_partial(false);
    let result = module.load_from(&mut store).wrap_err_with(|| {
        format!(
            "failed to import vocoder state dictionary {}",
            path.display()
        )
    })?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "vocoder state dictionary {} did not match the native module (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

/// Load a prepared Burnpack containing the vocoder.
///
/// # Errors
///
/// Returns an error if the Burnpack cannot be read or does not fully populate
/// the supplied vocoder.
pub fn load_hifigan_burnpack<B: Backend>(module: &mut HiFiGan<B>, path: &Path) -> eyre::Result<()> {
    let mut store = BurnpackStore::from_file(path).allow_partial(false);
    let result = module
        .load_from(&mut store)
        .wrap_err_with(|| format!("failed to load vocoder Burnpack {}", path.display()))?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "vocoder Burnpack {} did not match the native module (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}
