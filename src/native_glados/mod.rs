#![expect(
    clippy::disallowed_types,
    reason = "Burn's Module derive uses serde internally for model records."
)]

#[cfg(all(
    feature = "cuda",
    not(any(
        feature = "burn-cuda-fused",
        feature = "burn-wgpu",
        feature = "burn-vulkan"
    ))
))]
mod cuda_lstm;
#[cfg(feature = "burn-tch")]
mod tch_cbhg;
#[cfg(feature = "burn-tch")]
mod tch_gru;
#[cfg(feature = "burn-tch")]
mod tch_lstm;
#[cfg(feature = "burn-tch")]
mod tch_predictor;
#[cfg(feature = "torchscript")]
pub mod torchscript;
pub mod vocoder;

use burn::config::Config;
use burn::module::Module;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Embedding;
use burn::nn::EmbeddingConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::gru::Gru;
use burn::nn::gru::GruConfig;
use burn::nn::lstm::BiLstm;
use burn::nn::lstm::BiLstmConfig;
use burn::nn::lstm::Lstm;
use burn::nn::pool::MaxPool1d;
use burn::nn::pool::MaxPool1dConfig;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use burn_store::PytorchStore;
use eyre::WrapErr;
use eyre::bail;
use std::path::Path;
use std::time::Instant;

/// Backend used by the first native `GLaDOS` acoustic runtime.
pub type GladosCpuBackend = burn::backend::NdArray;

pub trait AcousticLstmBackend: Backend {
    fn forward_bidirectional_lstm(lstm: &BiLstm<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3>;

    fn forward_bidirectional_gru(
        gru: &BidirectionalGru<Self>,
        input: Tensor<Self, 3>,
    ) -> Tensor<Self, 3> {
        gru.forward(input)
    }

    fn forward_cbhg(cbhg: &Cbhg<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        cbhg.forward(input)
    }

    fn forward_series_predictor(
        predictor: &SeriesPredictor<Self>,
        tokens: Tensor<Self, 2, Int>,
        speaker_embedding: Tensor<Self, 2>,
        alpha: f32,
    ) -> Tensor<Self, 3> {
        forward_series_predictor_reference(predictor, tokens, speaker_embedding, alpha)
    }

    fn forward_conditional_series_predictor(
        predictor: &ConditionalSeriesPredictor<Self>,
        tokens: Tensor<Self, 2, Int>,
        pitch_conditions: Tensor<Self, 2, Int>,
        speaker_embedding: Tensor<Self, 2>,
        alpha: f32,
    ) -> Tensor<Self, 3> {
        forward_conditional_series_predictor_reference(
            predictor,
            tokens,
            pitch_conditions,
            speaker_embedding,
            alpha,
        )
    }
}

fn forward_bidirectional_lstm_reference<B: Backend>(
    lstm: &BiLstm<B>,
    input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let forward = forward_lstm_direction_reference(&lstm.forward, input.clone());
    let reverse = forward_lstm_direction_reference(&lstm.reverse, input.flip([1])).flip([1]);
    Tensor::cat(vec![forward, reverse], 2)
}

impl AcousticLstmBackend for GladosCpuBackend {
    fn forward_bidirectional_lstm(lstm: &BiLstm<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        forward_bidirectional_lstm_reference(lstm, input)
    }
}

#[cfg(feature = "cuda")]
impl AcousticLstmBackend for burn::backend::Cuda {
    fn forward_bidirectional_lstm(lstm: &BiLstm<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        #[cfg(any(
            feature = "burn-cuda-fused",
            feature = "burn-wgpu",
            feature = "burn-vulkan"
        ))]
        {
            forward_bidirectional_lstm_reference(lstm, input)
        }

        #[cfg(not(any(
            feature = "burn-cuda-fused",
            feature = "burn-wgpu",
            feature = "burn-vulkan"
        )))]
        {
            cuda_lstm::forward_bidirectional_lstm(lstm, input)
        }
    }
}

#[cfg(feature = "burn-tch")]
impl AcousticLstmBackend for burn::backend::LibTorch<f32> {
    fn forward_bidirectional_lstm(lstm: &BiLstm<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tch_lstm::forward_bidirectional_lstm(lstm, input)
    }

    fn forward_bidirectional_gru(
        gru: &BidirectionalGru<Self>,
        input: Tensor<Self, 3>,
    ) -> Tensor<Self, 3> {
        tch_gru::forward_bidirectional_gru(gru, input)
    }

    fn forward_cbhg(cbhg: &Cbhg<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        tch_cbhg::forward_cbhg(cbhg, input)
    }

    fn forward_series_predictor(
        predictor: &SeriesPredictor<Self>,
        tokens: Tensor<Self, 2, Int>,
        speaker_embedding: Tensor<Self, 2>,
        alpha: f32,
    ) -> Tensor<Self, 3> {
        tch_predictor::forward_series_predictor(predictor, tokens, speaker_embedding, alpha)
    }

    fn forward_conditional_series_predictor(
        predictor: &ConditionalSeriesPredictor<Self>,
        tokens: Tensor<Self, 2, Int>,
        pitch_conditions: Tensor<Self, 2, Int>,
        speaker_embedding: Tensor<Self, 2>,
        alpha: f32,
    ) -> Tensor<Self, 3> {
        tch_predictor::forward_conditional_series_predictor(
            predictor,
            tokens,
            pitch_conditions,
            speaker_embedding,
            alpha,
        )
    }
}

#[cfg(feature = "burn-wgpu")]
impl AcousticLstmBackend for burn::backend::Wgpu {
    fn forward_bidirectional_lstm(lstm: &BiLstm<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        forward_bidirectional_lstm_reference(lstm, input)
    }
}

#[cfg(feature = "burn-vulkan")]
impl AcousticLstmBackend for burn::backend::Vulkan {
    fn forward_bidirectional_lstm(lstm: &BiLstm<Self>, input: Tensor<Self, 3>) -> Tensor<Self, 3> {
        forward_bidirectional_lstm_reference(lstm, input)
    }
}

/// Backend used by the production vocoder runtime.
///
/// CUDA is enabled by default because the upstream application selects CUDA
/// on NVIDIA systems and the CPU `NdArray` backend is unsuitable for interactive
/// synthesis. Build with `--no-default-features` to use the portable CPU path.
#[cfg(feature = "cuda")]
pub type GladosVocoderBackend = burn::backend::Cuda;

/// Portable CPU fallback used when CUDA is disabled.
#[cfg(not(feature = "cuda"))]
pub type GladosVocoderBackend = GladosCpuBackend;

pub const GLADOS_TOKEN_COUNT: usize = 135;
pub const GLADOS_TOKEN_EMBEDDING_DIMENSION: usize = 256;
pub const GLADOS_SPEAKER_EMBEDDING_DIMENSION: usize = 256;

/// Load one upstream speaker embedding from the native little-endian artifact
/// emitted by `tools/export_glados_voices.py`.
///
/// # Errors
///
/// Returns an error when the artifact is not exactly one 256-element
/// float32 embedding or cannot be read.
///
/// # Panics
///
/// This function does not panic for a valid artifact; the byte-width check
/// guarantees that every conversion chunk has exactly four bytes.
pub fn load_voice_embedding<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> eyre::Result<Tensor<B, 2>> {
    let bytes = std::fs::read(path).map_err(|error| {
        eyre::eyre!("failed to read voice embedding {}: {error}", path.display())
    })?;
    let expected_bytes = GLADOS_SPEAKER_EMBEDDING_DIMENSION * size_of::<f32>();
    if bytes.len() != expected_bytes {
        bail!(
            "voice embedding {} has {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected_bytes
        );
    }
    let values = bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let array = <[u8; 4]>::try_from(chunk).map_err(|error| {
                eyre::eyre!("voice embedding contains a partial float32: {error}")
            })?;
            Ok(f32::from_le_bytes(array))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    Ok(Tensor::from_data(
        burn::tensor::TensorData::new(values, [1, GLADOS_SPEAKER_EMBEDDING_DIMENSION]),
        device,
    ))
}

/// Configuration shared by the convolutional predictor family in `ForwardTacotron`.
#[derive(Config, Debug)]
pub struct SeriesPredictorConfig {
    pub token_count: usize,
    pub token_embedding_dimension: usize,
    pub speaker_embedding_dimension: usize,
    pub convolution_count: usize,
    pub convolution_channels: usize,
    pub convolution_kernel_size: usize,
    pub recurrent_hidden_dimension: usize,
    pub output_dimension: usize,
}

impl SeriesPredictorConfig {
    fn init_convolutions<B: Backend>(
        &self,
        input_channels: usize,
        device: &B::Device,
    ) -> Vec<ConvNormBlock<B>> {
        (0..self.convolution_count)
            .map(|index| {
                ConvNormBlockConfig::new(
                    if index == 0 {
                        input_channels
                    } else {
                        self.convolution_channels
                    },
                    self.convolution_channels,
                    self.convolution_kernel_size,
                )
                .init(device)
            })
            .collect()
    }

    /// Initialize a predictor that does not consume pitch-condition tokens.
    #[must_use]
    pub fn init_series<B: Backend>(&self, device: &B::Device) -> SeriesPredictor<B> {
        SeriesPredictor {
            embedding: EmbeddingConfig::new(self.token_count, self.token_embedding_dimension)
                .init(device),
            convs: self.init_convolutions(
                self.token_embedding_dimension + self.speaker_embedding_dimension,
                device,
            ),
            rnn: BidirectionalGruConfig::new(
                self.convolution_channels,
                self.recurrent_hidden_dimension,
                true,
            )
            .init::<B>(device),
            lin: LinearConfig::new(self.recurrent_hidden_dimension * 2, self.output_dimension)
                .init(device),
        }
    }

    /// Initialize a predictor that consumes pitch-condition tokens.
    #[must_use]
    pub fn init_conditional<B: Backend>(
        &self,
        pitch_condition_count: usize,
        pitch_condition_dimension: usize,
        device: &B::Device,
    ) -> ConditionalSeriesPredictor<B> {
        ConditionalSeriesPredictor {
            embedding: EmbeddingConfig::new(self.token_count, self.token_embedding_dimension)
                .init(device),
            pitch_cond_embedding: EmbeddingConfig::new(
                pitch_condition_count,
                pitch_condition_dimension,
            )
            .init(device),
            convs: self.init_convolutions(
                self.token_embedding_dimension
                    + pitch_condition_dimension
                    + self.speaker_embedding_dimension,
                device,
            ),
            rnn: BidirectionalGruConfig::new(
                self.convolution_channels,
                self.recurrent_hidden_dimension,
                true,
            )
            .init::<B>(device),
            lin: LinearConfig::new(self.recurrent_hidden_dimension * 2, self.output_dimension)
                .init(device),
        }
    }
}

#[derive(Config, Debug)]
struct BidirectionalGruConfig {
    input_dimension: usize,
    hidden_dimension: usize,
    bias: bool,
}

impl BidirectionalGruConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> BidirectionalGru<B> {
        BidirectionalGru {
            forward: GruConfig::new(self.input_dimension, self.hidden_dimension, self.bias)
                .init(device),
            reverse: GruConfig::new(self.input_dimension, self.hidden_dimension, self.bias)
                .init(device),
        }
    }
}

/// A bidirectional GRU with the same reset-after gate behavior as `PyTorch`.
#[derive(Module, Debug)]
pub struct BidirectionalGru<B: Backend> {
    pub forward: Gru<B>,
    pub reverse: Gru<B>,
}

impl<B: Backend> BidirectionalGru<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let forward = Self::forward_direction(&self.forward, input.clone());
        let reverse = Self::forward_direction(&self.reverse, input.flip([1])).flip([1]);
        Tensor::cat(vec![forward, reverse], 2)
    }

    /// Execute one GRU direction with the three gate projections packed.
    ///
    /// Burn's generic `Gru` implementation performs separate input and hidden
    /// matrix multiplications for each of update, reset, and new gates at every
    /// timestep. The model uses PyTorch-compatible reset-after GRUs, so the
    /// three input projections and three hidden projections can be concatenated
    /// into two matrix multiplications without changing the serialized module
    /// layout. Keeping the original `Gru` fields also preserves Burnpack
    /// loading compatibility.
    fn forward_direction(gru: &Gru<B>, batched_input: Tensor<B, 3>) -> Tensor<B, 3> {
        if !gru.reset_after {
            return gru.forward(batched_input, None);
        }

        let device = batched_input.device();
        let [batch_size, sequence_length, _] = batched_input.dims();
        let hidden_channels = gru.d_hidden;
        let input_weight = Tensor::cat(
            vec![
                gru.update_gate.input_transform.weight.val(),
                gru.reset_gate.input_transform.weight.val(),
                gru.new_gate.input_transform.weight.val(),
            ],
            1,
        );
        let hidden_weight = Tensor::cat(
            vec![
                gru.update_gate.hidden_transform.weight.val(),
                gru.reset_gate.hidden_transform.weight.val(),
                gru.new_gate.hidden_transform.weight.val(),
            ],
            1,
        );
        let input_bias = concat_gru_biases(
            gru.update_gate.input_transform.bias.as_ref(),
            gru.reset_gate.input_transform.bias.as_ref(),
            gru.new_gate.input_transform.bias.as_ref(),
        );
        let hidden_bias = concat_gru_biases(
            gru.update_gate.hidden_transform.bias.as_ref(),
            gru.reset_gate.hidden_transform.bias.as_ref(),
            gru.new_gate.hidden_transform.bias.as_ref(),
        );

        let mut hidden = Tensor::zeros([batch_size, hidden_channels], &device);
        let mut outputs = Vec::with_capacity(sequence_length);
        for input_t in batched_input.iter_dim(1) {
            let input_t = input_t.squeeze_dim(1);
            let input_projection =
                add_gru_bias(input_t.matmul(input_weight.clone()), input_bias.as_ref());
            let hidden_projection = add_gru_bias(
                hidden.clone().matmul(hidden_weight.clone()),
                hidden_bias.as_ref(),
            );
            let update = burn::tensor::activation::sigmoid(
                input_projection
                    .clone()
                    .slice([0..batch_size, 0..hidden_channels])
                    + hidden_projection
                        .clone()
                        .slice([0..batch_size, 0..hidden_channels]),
            );
            let reset = burn::tensor::activation::sigmoid(
                input_projection
                    .clone()
                    .slice([0..batch_size, hidden_channels..2 * hidden_channels])
                    + hidden_projection
                        .clone()
                        .slice([0..batch_size, hidden_channels..2 * hidden_channels]),
            );
            let candidate = (input_projection
                .slice([0..batch_size, 2 * hidden_channels..3 * hidden_channels])
                + reset
                    * hidden_projection
                        .slice([0..batch_size, 2 * hidden_channels..3 * hidden_channels]))
            .tanh();
            hidden =
                candidate.mul(update.clone().sub_scalar(1).mul_scalar(-1)) + update.mul(hidden);
            outputs.push(hidden.clone().unsqueeze_dim(1));
        }

        Tensor::cat(outputs, 1)
    }
}

fn concat_gru_biases<B: Backend>(
    first: Option<&burn::module::Param<Tensor<B, 1>>>,
    second: Option<&burn::module::Param<Tensor<B, 1>>>,
    third: Option<&burn::module::Param<Tensor<B, 1>>>,
) -> Option<Tensor<B, 1>> {
    match (first, second, third) {
        (Some(first), Some(second), Some(third)) => {
            Some(Tensor::cat(vec![first.val(), second.val(), third.val()], 0))
        }
        (None, None, None) => None,
        _ => panic!("GRU gate biases must be consistently present"),
    }
}

fn add_gru_bias<B: Backend>(projection: Tensor<B, 2>, bias: Option<&Tensor<B, 1>>) -> Tensor<B, 2> {
    match bias {
        Some(bias) => projection + bias.clone().unsqueeze(),
        None => projection,
    }
}

#[derive(Config, Debug)]
struct ConvNormBlockConfig {
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
}

impl ConvNormBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConvNormBlock<B> {
        ConvNormBlock {
            conv: Conv1dConfig::new(self.input_channels, self.output_channels, self.kernel_size)
                .with_padding(PaddingConfig1d::Explicit(self.kernel_size / 2))
                .with_bias(false)
                .init(device),
            bnorm: BatchNormConfig::new(self.output_channels).init(device),
        }
    }

    fn init_no_relu<B: Backend>(&self, device: &B::Device) -> ConvNormBlockNoRelu<B> {
        ConvNormBlockNoRelu {
            conv: Conv1dConfig::new(self.input_channels, self.output_channels, self.kernel_size)
                .with_padding(PaddingConfig1d::Explicit(self.kernel_size / 2))
                .with_bias(false)
                .init(device),
            bnorm: BatchNormConfig::new(self.output_channels).init(device),
        }
    }
}

/// A `ForwardTacotron` convolution followed by batch normalization and `ReLU`.
#[derive(Module, Debug)]
pub struct ConvNormBlock<B: Backend> {
    pub conv: Conv1d<B>,
    pub bnorm: BatchNorm<B>,
}

impl<B: Backend> ConvNormBlock<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.bnorm.forward(relu(self.conv.forward(input)))
    }
}

#[derive(Module, Debug)]
pub(crate) struct ConvNormBlockNoRelu<B: Backend> {
    pub(crate) conv: Conv1d<B>,
    pub(crate) bnorm: BatchNorm<B>,
}

impl<B: Backend> ConvNormBlockNoRelu<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.bnorm.forward(self.conv.forward(input))
    }
}

#[derive(Config, Debug)]
struct HighwayConfig {
    dimension: usize,
}

impl HighwayConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Highway<B> {
        Highway {
            w1: LinearConfig::new(self.dimension, self.dimension).init(device),
            w2: LinearConfig::new(self.dimension, self.dimension).init(device),
        }
    }
}

/// A `ForwardTacotron` Highway layer.
#[derive(Module, Debug)]
pub struct Highway<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
}

impl<B: Backend> Highway<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = sigmoid(self.w2.forward(input.clone()));
        let transformed = relu(self.w1.forward(input.clone()));
        gate.clone() * transformed + (gate.ones_like() - gate) * input
    }
}

#[derive(Config, Debug)]
struct CbhgConfig {
    input_channels: usize,
    output_channels: usize,
    bank_count: usize,
    highway_channels: usize,
}

impl CbhgConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Cbhg<B> {
        let conv1d_bank = (1..=self.bank_count)
            .map(|kernel_size| {
                ConvNormBlockConfig::new(self.input_channels, 256, kernel_size).init(device)
            })
            .collect();
        let maxpool = MaxPool1dConfig::new(2)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init();
        let conv_project1 = ConvNormBlockConfig::new(self.bank_count * 256, 256, 3).init(device);
        let conv_project2 =
            ConvNormBlockConfig::new(256, self.output_channels, 3).init_no_relu(device);
        let pre_highway = LinearConfig::new(self.output_channels, self.highway_channels)
            .with_bias(false)
            .init(device);
        let highways = (0..4)
            .map(|_| HighwayConfig::new(self.highway_channels).init(device))
            .collect();
        let rnn = BidirectionalGruConfig::new(self.highway_channels, 256, true).init(device);

        Cbhg {
            conv1d_bank,
            maxpool,
            conv_project1,
            conv_project2,
            pre_highway,
            highways,
            rnn,
        }
    }
}

/// A CBHG block used by the upstream `ForwardTacotron` graph.
#[derive(Module, Debug)]
pub struct Cbhg<B: Backend> {
    pub conv1d_bank: Vec<ConvNormBlock<B>>,
    pub maxpool: MaxPool1d,
    pub conv_project1: ConvNormBlock<B>,
    pub(crate) conv_project2: ConvNormBlockNoRelu<B>,
    pub pre_highway: Linear<B>,
    pub highways: Vec<Highway<B>>,
    pub rnn: BidirectionalGru<B>,
}

impl<B: Backend> Cbhg<B> {
    /// Run the CBHG block on `[batch, channels, sequence]` input.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, sequence_length] = input.dims();
        let bank = self
            .conv1d_bank
            .iter()
            .map(|conv| {
                let output = conv.forward(input.clone());
                let [batch_size, channels, _] = output.dims();
                output.slice([0..batch_size, 0..channels, 0..sequence_length])
            })
            .collect::<Vec<_>>();
        let bank = Tensor::cat(bank, 1);
        let pooled = self.maxpool.forward(bank);
        let [batch_size, channels, _] = pooled.dims();
        let pooled = pooled.slice([0..batch_size, 0..channels, 0..sequence_length]);
        let projected = self
            .conv_project2
            .forward(self.conv_project1.forward(pooled));
        let mut output = self.pre_highway.forward((projected + input).transpose());
        for highway in &self.highways {
            output = highway.forward(output);
        }
        self.rnn.forward(output)
    }
}

/// Stateless duration expansion used between the predictors and acoustic LSTM.
#[derive(Clone, Module, Debug)]
pub struct LengthRegulator;

const DURATION_TIE_EPSILON: f32 = 0.004;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Model durations are finite, non-negative, and bounded frame counts."
)]
fn duration_to_frame_count(duration: f32) -> usize {
    if !duration.is_finite() {
        return 0;
    }
    let duration = duration.max(0.0);
    let lower = duration.floor();
    let fractional = duration - lower;
    let rounded = if (fractional - 0.5).abs() <= DURATION_TIE_EPSILON {
        if (lower as usize).is_multiple_of(2) {
            lower
        } else {
            lower + 1.0
        }
    } else {
        (duration + 0.5).floor()
    };
    rounded as usize
}

impl LengthRegulator {
    fn forward<B: Backend>(input: &Tensor<B, 3>, durations: &Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, token_count, channels] = input.dims();
        let duration_values = durations
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("duration tensor should contain f32 values");
        let mut rows = Vec::with_capacity(batch_size);
        let mut lengths = Vec::with_capacity(batch_size);

        for batch_index in 0..batch_size {
            let mut frames = Vec::new();
            let mut length = 0;
            for token_index in 0..token_count {
                let duration = duration_values[batch_index * token_count + token_index];
                let repeat_count = duration_to_frame_count(duration);
                if repeat_count == 0 {
                    continue;
                }
                let token = input
                    .clone()
                    .slice([
                        batch_index..batch_index + 1,
                        token_index..token_index + 1,
                        0..channels,
                    ])
                    .repeat(&[1, repeat_count, 1]);
                frames.push(token);
                length += repeat_count;
            }
            if frames.is_empty() {
                rows.push(Tensor::zeros([1, 1, channels], &input.device()));
                lengths.push(1);
            } else {
                rows.push(Tensor::cat(frames, 1));
                lengths.push(length);
            }
        }

        let max_length = lengths.iter().copied().max().unwrap_or(1);
        let device = input.device();
        let rows = rows
            .into_iter()
            .zip(lengths)
            .map(|(row, length)| {
                let mut padded = Tensor::zeros([1, max_length, channels], &device);
                padded = padded.slice_assign([0..1, 0..length, 0..channels], row);
                padded
            })
            .collect();
        Tensor::cat(rows, 0)
    }
}

/// Outputs produced by the native acoustic `ForwardTacotron` graph.
#[derive(Debug)]
pub struct GladosMelOutput<B: Backend> {
    pub mel: Tensor<B, 3>,
    pub mel_post: Tensor<B, 3>,
    pub postnet: Tensor<B, 3>,
    pub conditioning: Tensor<B, 3>,
    pub regulated: Tensor<B, 3>,
    pub lstm_output: Tensor<B, 3>,
    pub pitch_projection: Tensor<B, 3>,
    pub energy_projection: Tensor<B, 3>,
    pub durations: Tensor<B, 2>,
    pub pitch: Tensor<B, 3>,
    pub energy: Tensor<B, 3>,
    pub pitch_conditions: Tensor<B, 3, Int>,
}

/// Intermediate acoustic tensors used by parity diagnostics.
#[derive(Debug)]
pub struct GladosAcousticTrace<B: Backend> {
    pub prenet: Tensor<B, 3>,
    pub base_conditioning: Tensor<B, 3>,
    pub pitch_projection: Tensor<B, 3>,
    pub energy_projection: Tensor<B, 3>,
    pub conditioning: Tensor<B, 3>,
    pub regulated: Tensor<B, 3>,
    pub lstm_output: Tensor<B, 3>,
    pub mel: Tensor<B, 3>,
    pub postnet: Tensor<B, 3>,
    pub mel_post: Tensor<B, 3>,
}

/// The pure-Rust acoustic portion of the upstream `GLaDOS` model.
#[derive(Module, Debug)]
pub struct GladosAcousticModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub dur_pred: ConditionalSeriesPredictor<B>,
    pub pitch_cond_pred: SeriesPredictor<B>,
    pub pitch_pred: ConditionalSeriesPredictor<B>,
    pub energy_pred: SeriesPredictor<B>,
    pub pitch_proj: Conv1d<B>,
    pub energy_proj: Conv1d<B>,
    pub prenet: Cbhg<B>,
    pub lstm: BiLstm<B>,
    pub lin: Linear<B>,
    pub postnet: Cbhg<B>,
    pub post_proj: Linear<B>,
    pub lr: LengthRegulator,
}

impl<B: Backend> GladosAcousticModel<B> {
    /// Initialize the fixed architecture used by `glados-new.pt`.
    #[must_use]
    pub fn init(device: &B::Device) -> Self {
        let predictor = SeriesPredictorConfig::new(
            GLADOS_TOKEN_COUNT,
            128,
            GLADOS_SPEAKER_EMBEDDING_DIMENSION,
            3,
            256,
            5,
            128,
            1,
        );
        let dur_pred = predictor.init_conditional::<B>(4, 4, device);
        let pitch_cond_pred = SeriesPredictorConfig::new(
            GLADOS_TOKEN_COUNT,
            128,
            GLADOS_SPEAKER_EMBEDDING_DIMENSION,
            3,
            256,
            5,
            128,
            3,
        )
        .init_series::<B>(device);
        let pitch_pred = SeriesPredictorConfig::new(
            GLADOS_TOKEN_COUNT,
            128,
            GLADOS_SPEAKER_EMBEDDING_DIMENSION,
            3,
            256,
            5,
            256,
            1,
        )
        .init_conditional::<B>(4, 4, device);
        let energy_pred = SeriesPredictorConfig::new(
            GLADOS_TOKEN_COUNT,
            128,
            GLADOS_SPEAKER_EMBEDDING_DIMENSION,
            3,
            256,
            5,
            64,
            1,
        )
        .init_series::<B>(device);

        Self {
            embedding: EmbeddingConfig::new(GLADOS_TOKEN_COUNT, GLADOS_TOKEN_EMBEDDING_DIMENSION)
                .init(device),
            dur_pred,
            pitch_cond_pred,
            pitch_pred,
            energy_pred,
            pitch_proj: Conv1dConfig::new(1, 768, 3)
                .with_padding(PaddingConfig1d::Explicit(1))
                .init(device),
            energy_proj: Conv1dConfig::new(1, 768, 3)
                .with_padding(PaddingConfig1d::Explicit(1))
                .init(device),
            prenet: CbhgConfig::new(256, 256, 16, 256).init(device),
            lstm: BiLstmConfig::new(768, 512, true).init(device),
            lin: LinearConfig::new(1024, 80).init(device),
            postnet: CbhgConfig::new(80, 80, 8, 256).init(device),
            post_proj: LinearConfig::new(512, 80).with_bias(false).init(device),
            lr: LengthRegulator,
        }
    }

    /// Generate mel spectrograms from prepared token IDs and a speaker vector.
    ///
    /// # Panics
    ///
    /// Panics when more than one utterance is supplied; the first native
    /// runtime intentionally supports batch size one only.
    pub fn generate(
        &self,
        tokens: Tensor<B, 2, Int>,
        speaker_embedding: Tensor<B, 2>,
        alpha: f32,
    ) -> GladosMelOutput<B>
    where
        B: AcousticLstmBackend,
    {
        let started = Instant::now();
        let pitch_condition_scores =
            self.pitch_cond_pred
                .forward(tokens.clone(), speaker_embedding.clone(), 1.0);
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic pitch-condition prediction complete"
        );
        let [batch_size, _, _] = pitch_condition_scores.dims();
        assert_eq!(
            batch_size, 1,
            "the first native GLaDOS runtime supports one generated utterance at a time"
        );
        let pitch_conditions = pitch_condition_scores
            .squeeze_dim::<2>(0)
            .argmax(1)
            .squeeze_dim::<1>(1)
            .unsqueeze_dim::<2>(0);
        let durations = self
            .dur_pred
            .forward(
                tokens.clone(),
                pitch_conditions.clone(),
                speaker_embedding.clone(),
                alpha,
            )
            .squeeze_dim::<2>(2);
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic duration prediction complete"
        );
        let duration_values = durations
            .to_data()
            .to_vec::<f32>()
            .expect("duration tensor should contain f32 values");
        let duration_sum = duration_values.iter().sum::<f32>();
        let duration_rounding_boundaries = duration_rounding_boundaries(&duration_values);
        let duration_frame_count = if duration_sum <= 0.0 {
            duration_values.len().saturating_mul(2)
        } else {
            duration_frame_count_from_values(&duration_values)
        };
        tracing::info!(
            duration_sum,
            duration_frame_count,
            fallback = duration_sum <= 0.0,
            "acoustic duration schedule ready"
        );
        tracing::debug!(
            ?duration_rounding_boundaries,
            "acoustic duration rounding boundaries"
        );
        let durations = if duration_sum <= 0.0 {
            Tensor::full(durations.dims(), 2.0, &durations.device())
        } else {
            durations
        };
        let pitch = self
            .pitch_pred
            .forward(
                tokens.clone(),
                pitch_conditions.clone(),
                speaker_embedding.clone(),
                1.0,
            )
            .transpose();
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic pitch prediction complete"
        );
        let energy = self
            .energy_pred
            .forward(tokens.clone(), speaker_embedding.clone(), 1.0)
            .transpose();
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic energy prediction complete"
        );
        self.generate_mel(
            tokens,
            speaker_embedding,
            durations,
            pitch,
            energy,
            pitch_conditions,
        )
    }

    fn generate_mel(
        &self,
        tokens: Tensor<B, 2, Int>,
        speaker_embedding: Tensor<B, 2>,
        durations: Tensor<B, 2>,
        pitch: Tensor<B, 3>,
        energy: Tensor<B, 3>,
        pitch_conditions: Tensor<B, 2, Int>,
    ) -> GladosMelOutput<B>
    where
        B: AcousticLstmBackend,
    {
        let started = Instant::now();
        let trace = self.trace_mel(
            tokens,
            speaker_embedding,
            &durations,
            pitch.clone(),
            energy.clone(),
        );
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic mel graph complete"
        );
        GladosMelOutput {
            mel: trace.mel,
            mel_post: trace.mel_post,
            postnet: trace.postnet,
            conditioning: trace.conditioning,
            regulated: trace.regulated,
            lstm_output: trace.lstm_output,
            pitch_projection: trace.pitch_projection,
            energy_projection: trace.energy_projection,
            durations,
            pitch,
            energy,
            pitch_conditions: pitch_conditions.unsqueeze_dim::<3>(1),
        }
    }

    /// Run the acoustic graph and retain its intermediate tensors for parity checks.
    pub fn trace_mel(
        &self,
        tokens: Tensor<B, 2, Int>,
        speaker_embedding: Tensor<B, 2>,
        durations: &Tensor<B, 2>,
        pitch: Tensor<B, 3>,
        energy: Tensor<B, 3>,
    ) -> GladosAcousticTrace<B>
    where
        B: AcousticLstmBackend,
    {
        let started = Instant::now();
        let embedded = self.embedding.forward(tokens);
        let prenet = B::forward_cbhg(&self.prenet, embedded.transpose());
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic prenet complete"
        );
        let [_, sequence_length, _] = prenet.dims();
        let speaker_embedding = speaker_embedding
            .unsqueeze_dim(1)
            .repeat(&[1, sequence_length, 1]);
        let base_conditioning = Tensor::cat(vec![prenet.clone(), speaker_embedding], 2);
        let pitch_projection = self.pitch_proj.forward(pitch).transpose().mul_scalar(1.0);
        let energy_projection = self.energy_proj.forward(energy).transpose().mul_scalar(1.0);
        let conditioning =
            base_conditioning.clone() + pitch_projection.clone() + energy_projection.clone();
        let regulated = LengthRegulator::forward(&conditioning, durations);
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic length regulation complete"
        );
        let lstm_started = Instant::now();
        let hidden = B::forward_bidirectional_lstm(&self.lstm, regulated.clone());
        tracing::info!(
            elapsed_ms = lstm_started.elapsed().as_millis(),
            mel_frames = hidden.dims()[1],
            "acoustic bidirectional LSTM complete"
        );
        let mel_projection_started = Instant::now();
        let mel = self.lin.forward(hidden.clone()).transpose();
        tracing::debug!(
            elapsed_ms = mel_projection_started.elapsed().as_millis(),
            "acoustic mel projection complete"
        );
        let postnet = B::forward_cbhg(&self.postnet, mel.clone());
        let mel_post = self.post_proj.forward(postnet.clone()).transpose();
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "acoustic postnet complete"
        );
        GladosAcousticTrace {
            prenet,
            base_conditioning,
            pitch_projection,
            energy_projection,
            conditioning,
            regulated,
            lstm_output: hidden,
            mel,
            postnet,
            mel_post,
        }
    }
}

/// Execute both LSTM directions with their four gate projections packed.
///
/// This is the portable Burn implementation used as the correctness fallback
/// for every backend except the non-fused CUDA extension. Burn's generic LSTM
/// implementation performs one input and one hidden matrix multiplication for
/// each gate at every timestep. Packing the four gates keeps the Burnpack
/// module layout unchanged while reducing that to one input and one hidden
/// projection per timestep.
pub(super) fn forward_lstm_direction_reference<B: Backend>(
    lstm: &Lstm<B>,
    batched_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let device = batched_input.device();
    let [batch_size, sequence_length, _] = batched_input.dims();
    let hidden_channels = lstm.d_hidden;
    let input_weight = Tensor::cat(
        vec![
            lstm.input_gate.input_transform.weight.val(),
            lstm.forget_gate.input_transform.weight.val(),
            lstm.output_gate.input_transform.weight.val(),
            lstm.cell_gate.input_transform.weight.val(),
        ],
        1,
    );
    let hidden_weight = Tensor::cat(
        vec![
            lstm.input_gate.hidden_transform.weight.val(),
            lstm.forget_gate.hidden_transform.weight.val(),
            lstm.output_gate.hidden_transform.weight.val(),
            lstm.cell_gate.hidden_transform.weight.val(),
        ],
        1,
    );
    let input_bias = concat_lstm_biases(
        lstm.input_gate.input_transform.bias.as_ref(),
        lstm.forget_gate.input_transform.bias.as_ref(),
        lstm.output_gate.input_transform.bias.as_ref(),
        lstm.cell_gate.input_transform.bias.as_ref(),
    );
    let hidden_bias = concat_lstm_biases(
        lstm.input_gate.hidden_transform.bias.as_ref(),
        lstm.forget_gate.hidden_transform.bias.as_ref(),
        lstm.output_gate.hidden_transform.bias.as_ref(),
        lstm.cell_gate.hidden_transform.bias.as_ref(),
    );

    let mut cell = Tensor::zeros([batch_size, hidden_channels], &device);
    let mut hidden = Tensor::zeros([batch_size, hidden_channels], &device);
    let mut outputs = Vec::with_capacity(sequence_length);
    for input_t in batched_input.iter_dim(1) {
        let input_t = input_t.squeeze_dim(1);
        let input_projection =
            add_projection_bias(input_t.matmul(input_weight.clone()), input_bias.as_ref());
        let hidden_projection = add_projection_bias(
            hidden.clone().matmul(hidden_weight.clone()),
            hidden_bias.as_ref(),
        );
        let input_values = sigmoid(
            input_projection
                .clone()
                .slice([0..batch_size, 0..hidden_channels])
                + hidden_projection
                    .clone()
                    .slice([0..batch_size, 0..hidden_channels]),
        );
        let forget_values = sigmoid(
            input_projection
                .clone()
                .slice([0..batch_size, hidden_channels..2 * hidden_channels])
                + hidden_projection
                    .clone()
                    .slice([0..batch_size, hidden_channels..2 * hidden_channels]),
        );
        let output_values = sigmoid(
            input_projection
                .clone()
                .slice([0..batch_size, 2 * hidden_channels..3 * hidden_channels])
                + hidden_projection
                    .clone()
                    .slice([0..batch_size, 2 * hidden_channels..3 * hidden_channels]),
        );
        let candidate = (input_projection
            .slice([0..batch_size, 3 * hidden_channels..4 * hidden_channels])
            + hidden_projection.slice([0..batch_size, 3 * hidden_channels..4 * hidden_channels]))
        .tanh();

        cell = forget_values * cell + input_values * candidate;
        hidden = output_values * cell.clone().tanh();
        outputs.push(hidden.clone().unsqueeze_dim(1));
    }

    Tensor::cat(outputs, 1)
}

fn concat_lstm_biases<B: Backend>(
    first: Option<&burn::module::Param<Tensor<B, 1>>>,
    second: Option<&burn::module::Param<Tensor<B, 1>>>,
    third: Option<&burn::module::Param<Tensor<B, 1>>>,
    fourth: Option<&burn::module::Param<Tensor<B, 1>>>,
) -> Option<Tensor<B, 1>> {
    match (first, second, third, fourth) {
        (Some(first), Some(second), Some(third), Some(fourth)) => Some(Tensor::cat(
            vec![first.val(), second.val(), third.val(), fourth.val()],
            0,
        )),
        (None, None, None, None) => None,
        _ => panic!("LSTM gate biases must be consistently present"),
    }
}

fn add_projection_bias<B: Backend>(
    projection: Tensor<B, 2>,
    bias: Option<&Tensor<B, 1>>,
) -> Tensor<B, 2> {
    match bias {
        Some(bias) => projection + bias.clone().unsqueeze(),
        None => projection,
    }
}

fn duration_frame_count_from_values(values: &[f32]) -> usize {
    values.iter().copied().map(duration_to_frame_count).sum()
}

fn duration_rounding_boundaries(values: &[f32]) -> Vec<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| {
            let fractional = value - value.floor();
            (fractional - 0.5)
                .abs()
                .le(&DURATION_TIE_EPSILON)
                .then_some((index, value))
        })
        .collect()
}

/// A predictor used for pitch-condition and energy prediction.
#[derive(Module, Debug)]
pub struct SeriesPredictor<B: Backend> {
    pub embedding: Embedding<B>,
    pub convs: Vec<ConvNormBlock<B>>,
    pub rnn: BidirectionalGru<B>,
    pub lin: Linear<B>,
}

fn forward_series_predictor_reference<B: AcousticLstmBackend>(
    predictor: &SeriesPredictor<B>,
    tokens: Tensor<B, 2, Int>,
    speaker_embedding: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 3> {
    let embedded = predictor.embedding.forward(tokens);
    let [_, sequence_length, _] = embedded.dims();
    let speaker_embedding = speaker_embedding
        .unsqueeze_dim(1)
        .repeat(&[1, sequence_length, 1]);
    let mut input = Tensor::cat(vec![embedded, speaker_embedding], 2).transpose();
    for conv in &predictor.convs {
        input = conv.forward(input);
    }
    let input = B::forward_bidirectional_gru(&predictor.rnn, input.transpose());
    predictor.lin.forward(input).div_scalar(alpha)
}

impl<B: Backend> SeriesPredictor<B> {
    /// Run the predictor on token IDs and speaker embeddings.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        speaker_embedding: Tensor<B, 2>,
        alpha: f32,
    ) -> Tensor<B, 3>
    where
        B: AcousticLstmBackend,
    {
        B::forward_series_predictor(self, tokens, speaker_embedding, alpha)
    }
}

/// A predictor used for duration and pitch prediction.
#[derive(Module, Debug)]
pub struct ConditionalSeriesPredictor<B: Backend> {
    pub embedding: Embedding<B>,
    pub pitch_cond_embedding: Embedding<B>,
    pub convs: Vec<ConvNormBlock<B>>,
    pub rnn: BidirectionalGru<B>,
    pub lin: Linear<B>,
}

fn forward_conditional_series_predictor_reference<B: AcousticLstmBackend>(
    predictor: &ConditionalSeriesPredictor<B>,
    tokens: Tensor<B, 2, Int>,
    pitch_conditions: Tensor<B, 2, Int>,
    speaker_embedding: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 3> {
    let embedded = predictor.embedding.forward(tokens);
    let pitch_conditions = predictor.pitch_cond_embedding.forward(pitch_conditions);
    let [_, sequence_length, _] = embedded.dims();
    let speaker_embedding = speaker_embedding
        .unsqueeze_dim(1)
        .repeat(&[1, sequence_length, 1]);
    let mut input = Tensor::cat(vec![embedded, pitch_conditions, speaker_embedding], 2).transpose();
    for conv in &predictor.convs {
        input = conv.forward(input);
    }
    let input = B::forward_bidirectional_gru(&predictor.rnn, input.transpose());
    predictor.lin.forward(input).div_scalar(alpha)
}

impl<B: Backend> ConditionalSeriesPredictor<B> {
    /// Run the predictor on token IDs, pitch-condition IDs, and speaker embeddings.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        pitch_conditions: Tensor<B, 2, Int>,
        speaker_embedding: Tensor<B, 2>,
        alpha: f32,
    ) -> Tensor<B, 3>
    where
        B: AcousticLstmBackend,
    {
        B::forward_conditional_series_predictor(
            self,
            tokens,
            pitch_conditions,
            speaker_embedding,
            alpha,
        )
    }
}

/// Load a Burnpack into a prepared predictor module.
///
/// The offline conversion step is responsible for splitting `PyTorch` GRU gate
/// tensors into Burn's gate modules. The runtime therefore only reads native
/// Burnpack weights and never needs Python or `TorchScript`.
///
/// # Errors
///
/// Returns an error if the Burnpack cannot be read or does not fully populate
/// the supplied module.
pub fn load_series_predictor_burnpack<B: Backend>(
    module: &mut SeriesPredictor<B>,
    path: &Path,
) -> eyre::Result<()> {
    let mut store = BurnpackStore::from_file(path).allow_partial(false);
    let result = module
        .load_from(&mut store)
        .wrap_err_with(|| format!("failed to load predictor Burnpack {}", path.display()))?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "predictor Burnpack {} did not match the native module (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

/// Import a converter-produced `PyTorch` state dictionary into a predictor.
///
/// The input is not an upstream `TorchScript` file. It is the deterministic
/// interchange file emitted by `tools/export_glados_predictor.py`, where
/// `PyTorch` GRU gates have already been split into Burn module fields.
///
/// # Errors
///
/// Returns an error if the state dictionary cannot be read or does not fully
/// populate the supplied module.
pub fn load_series_predictor_pytorch<B: Backend>(
    module: &mut SeriesPredictor<B>,
    path: &Path,
) -> eyre::Result<()> {
    load_predictor_pytorch(module, path)
}

/// Import a converter-produced `PyTorch` state dictionary into a conditional
/// predictor.
///
/// # Errors
///
/// Returns an error if the state dictionary cannot be read or does not fully
/// populate the supplied module.
pub fn load_conditional_series_predictor_pytorch<B: Backend>(
    module: &mut ConditionalSeriesPredictor<B>,
    path: &Path,
) -> eyre::Result<()> {
    load_predictor_pytorch(module, path)
}

/// Import a converter-produced full acoustic checkpoint into `GLaDOS`.
///
/// The checkpoint is emitted by `tools/export_glados_predictor.py --full` and
/// contains the native Burn names for the predictors, CBHG blocks, and
/// bidirectional LSTM.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be read or does not fully
/// populate the supplied model.
pub fn load_acoustic_model_pytorch<B: Backend>(
    module: &mut GladosAcousticModel<B>,
    path: &Path,
) -> eyre::Result<()> {
    load_predictor_pytorch(module, path)
}

/// Load a prepared Burnpack containing the full acoustic `GLaDOS` model.
///
/// # Errors
///
/// Returns an error if the Burnpack cannot be read or does not fully populate
/// the supplied model.
pub fn load_acoustic_model_burnpack<B: Backend>(
    module: &mut GladosAcousticModel<B>,
    path: &Path,
) -> eyre::Result<()> {
    let mut store = BurnpackStore::from_file(path).allow_partial(false);
    let result = module
        .load_from(&mut store)
        .wrap_err_with(|| format!("failed to load acoustic Burnpack {}", path.display()))?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "acoustic Burnpack {} did not match the native model (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

fn load_predictor_pytorch<B: Backend, M: ModuleSnapshot<B>>(
    module: &mut M,
    path: &Path,
) -> eyre::Result<()> {
    let mut store = PytorchStore::from_file(path).allow_partial(false);
    let result = module.load_from(&mut store).wrap_err_with(|| {
        format!(
            "failed to import predictor state dictionary {}",
            path.display()
        )
    })?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "predictor state dictionary {} did not match the native module (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    fn config() -> SeriesPredictorConfig {
        SeriesPredictorConfig::new(135, 128, 256, 3, 256, 5, 128, 1)
    }

    fn unique_path() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("teamy-tts-predictor-{nanos}.bpk"))
    }

    #[test]
    fn series_predictor_runs_on_ndarray_and_round_trips_burnpack() {
        type Backend = GladosCpuBackend;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let config = config();
        let module = config.init_series::<Backend>(&device);
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::from([[1_i32, 2, 3, 4]]), &device);
        let speaker = Tensor::<Backend, 2>::zeros([1, 256], &device);
        let output = module.forward(tokens, speaker, 1.0);
        assert_eq!(output.dims(), [1, 4, 1]);

        let path = unique_path();
        let mut store = BurnpackStore::from_file(&path).overwrite(true);
        module
            .save_into(&mut store)
            .expect("predictor should save as Burnpack");
        let mut loaded = config.init_series::<Backend>(&device);
        load_series_predictor_burnpack(&mut loaded, &path)
            .expect("predictor should load from Burnpack");
        std::fs::remove_file(path).expect("test Burnpack should be removable");
    }

    #[test]
    fn acoustic_model_initializes_native_architecture() {
        type Backend = GladosCpuBackend;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let model = GladosAcousticModel::<Backend>::init(&device);
        let tokens = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(vec![1_i32; GLADOS_TOKEN_COUNT], [1, GLADOS_TOKEN_COUNT]),
            &device,
        );
        let embedded = model.embedding.forward(tokens);
        assert_eq!(embedded.dims(), [1, GLADOS_TOKEN_COUNT, 256]);
    }

    #[test]
    fn duration_rounding_is_stable_at_backend_precision_boundaries() {
        assert_eq!(duration_to_frame_count(4.498), 4);
        assert_eq!(duration_to_frame_count(4.502), 4);
        assert_eq!(duration_to_frame_count(5.498), 6);
        assert_eq!(duration_to_frame_count(5.502), 6);
        assert_eq!(duration_to_frame_count(6.506), 7);
        assert_eq!(duration_to_frame_count(4.49), 4);
        assert_eq!(duration_to_frame_count(4.51), 5);
    }
}
