//! Native Burn implementation of the upstream `DeepPhonemizer` forward model.

#![expect(
    clippy::disallowed_types,
    reason = "Burn's Module derive uses serde internally for model records."
)]

use burn::module::Module;
use burn::module::Param;
use burn::nn::Embedding;
use burn::nn::EmbeddingConfig;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use burn_store::PytorchStore;
use eyre::WrapErr;
use eyre::bail;
use std::path::Path;

const D_MODEL: usize = 512;
const D_FF: usize = 1024;
const HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / HEADS;
// PyTorch MultiheadAttention scales by sqrt(head_dim), not head_dim.
const ATTENTION_SCALE: f32 = 11.313_708;
const LAYERS: usize = 6;
const TEXT_VOCAB: usize = 64;
const PHONEME_VOCAB: usize = 53;
const MAX_SEQUENCE: usize = 5000;

/// IPA symbols emitted by the upstream `DeepPhonemizer` checkpoint, in output
/// vocabulary order after pad/language/end special tokens.
pub const DEEP_PHONEMIZER_SYMBOLS: &str = "abdefghijklmnoprstuvwxyzæçðøŋœɐɑɔəɛɝɹɡɪʁʃʊʌʏʒʔː͡θ";

/// Small stage snapshots used by the development parity example.
#[derive(Debug)]
pub struct FrontendTrace {
    pub embedding: Vec<f32>,
    pub positional: Vec<f32>,
    pub layer0_query: Vec<f32>,
    pub layer0_key: Vec<f32>,
    pub layer0_value: Vec<f32>,
    pub layer0_weights: Vec<f32>,
    pub layer0_context: Vec<f32>,
    pub layer0_merged_context: Vec<f32>,
    pub layer0_attention: Vec<f32>,
    pub layer0_norm1: Vec<f32>,
    pub layer0_feed_forward: Vec<f32>,
    pub layers: Vec<Vec<f32>>,
    pub norm: Vec<f32>,
    pub logits: Vec<f32>,
}

#[derive(Module, Debug)]
struct FrontendPositionalEncoding<B: Backend> {
    scale: Param<Tensor<B, 1>>,
    pe: Param<Tensor<B, 3>>,
}

impl<B: Backend> FrontendPositionalEncoding<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            scale: Param::from_tensor(Tensor::ones([1], device)),
            pe: Param::from_tensor(Tensor::zeros([MAX_SEQUENCE, 1, D_MODEL], device)),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, sequence_length, _] = input.dims();
        let position = self
            .pe
            .val()
            .clone()
            .slice([0..sequence_length, 0..1, 0..D_MODEL])
            .swap_dims(0, 1);
        let scale = self
            .scale
            .val()
            .to_data()
            .to_vec::<f32>()
            .expect("positional scale should contain one f32")[0];
        input + position.mul_scalar(scale)
    }
}

#[derive(Module, Debug)]
struct FrontendAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> FrontendAttention<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            query: LinearConfig::new(D_MODEL, D_MODEL).init(device),
            key: LinearConfig::new(D_MODEL, D_MODEL).init(device),
            value: LinearConfig::new(D_MODEL, D_MODEL).init(device),
            out_proj: LinearConfig::new(D_MODEL, D_MODEL).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, sequence_length, _] = input.dims();
        let query = self
            .query
            .forward(input.clone())
            .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
            .swap_dims(1, 2);
        let key = self
            .key
            .forward(input.clone())
            .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
            .swap_dims(1, 2);
        let value = self
            .value
            .forward(input)
            .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
            .swap_dims(1, 2);
        let attention = softmax(query.matmul(key.transpose()).div_scalar(ATTENTION_SCALE), 3);
        self.out_proj
            .forward(attention.matmul(value).swap_dims(1, 2).reshape([
                batch_size,
                sequence_length,
                D_MODEL,
            ]))
    }
}

#[derive(Module, Debug)]
struct FrontendEncoderLayer<B: Backend> {
    self_attn: FrontendAttention<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> FrontendEncoderLayer<B> {
    fn init(device: &B::Device) -> Self {
        Self {
            self_attn: FrontendAttention::init(device),
            linear1: LinearConfig::new(D_MODEL, D_FF).init(device),
            linear2: LinearConfig::new(D_FF, D_MODEL).init(device),
            norm1: LayerNormConfig::new(D_MODEL).init(device),
            norm2: LayerNormConfig::new(D_MODEL).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let attention = self.self_attn.forward(input.clone());
        let input = self.norm1.forward(input + attention);
        let feed_forward = self
            .linear2
            .forward(relu(self.linear1.forward(input.clone())));
        self.norm2.forward(input + feed_forward)
    }
}

/// The non-autoregressive `DeepPhonemizer` Transformer used for unknown words.
#[derive(Module, Debug)]
pub struct GladosPhonemizer<B: Backend> {
    embedding: Embedding<B>,
    pos_encoder: FrontendPositionalEncoding<B>,
    encoder: Vec<FrontendEncoderLayer<B>>,
    norm: LayerNorm<B>,
    fc_out: Linear<B>,
}

impl<B: Backend> GladosPhonemizer<B> {
    /// Initialize the fixed upstream architecture.
    #[must_use]
    pub fn init(device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(TEXT_VOCAB, D_MODEL).init(device),
            pos_encoder: FrontendPositionalEncoding::init(device),
            encoder: (0..LAYERS)
                .map(|_| FrontendEncoderLayer::init(device))
                .collect(),
            norm: LayerNormConfig::new(D_MODEL).init(device),
            fc_out: LinearConfig::new(D_MODEL, PHONEME_VOCAB).init(device),
        }
    }

    /// Predict IPA for one dictionary-missing word.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported characters or a model output that
    /// does not contain any phonemes.
    pub fn phonemize_word(&self, word: &str) -> eyre::Result<String> {
        let indices = self.predict_token_ids(word)?;
        let mut output = String::new();
        let mut previous = None;
        for index in indices {
            if index == 3 {
                break;
            }
            if index == 0 || index == 1 || index == 2 {
                continue;
            }
            if previous == Some(index) {
                continue;
            }
            previous = Some(index);
            let Some(symbol_index) = usize::try_from(index - 4).ok() else {
                continue;
            };
            let Some(symbol) = DEEP_PHONEMIZER_SYMBOLS.chars().nth(symbol_index) else {
                continue;
            };
            output.push(symbol);
        }
        if output.is_empty() {
            bail!("phonemizer produced no phonemes for word {:?}", word);
        }
        Ok(output)
    }

    /// Return the greedy output vocabulary IDs for one word.
    ///
    /// This is exposed for parity tests; normal callers should use
    /// [`Self::phonemize_word`].
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported characters or a failed tensor copy.
    ///
    /// # Panics
    ///
    /// Panics only if the fixed phoneme vocabulary cannot fit in an `i64`.
    pub fn predict_token_ids(&self, word: &str) -> eyre::Result<Vec<i64>> {
        let logits = self.forward(self.input_for_word(word)?);
        let [_, sequence_length, _] = logits.dims();
        let mut indices = Vec::with_capacity(sequence_length);
        for position in 0..sequence_length {
            let row = logits
                .clone()
                .slice([0..1, position..position + 1, 0..PHONEME_VOCAB])
                .reshape([PHONEME_VOCAB])
                .to_data()
                .to_vec::<f32>()
                .map_err(|error| eyre::eyre!("failed to read phonemizer logits: {error:?}"))?;
            let mut best_index = 0_i64;
            let mut best_value = f32::NEG_INFINITY;
            for (index, value) in row.into_iter().enumerate() {
                if value > best_value {
                    best_value = value;
                    best_index = i64::try_from(index).expect("phoneme vocabulary fits i64");
                }
            }
            indices.push(best_index);
        }
        Ok(indices)
    }

    /// Trace the first sequence position through the native frontend model.
    ///
    /// This is intentionally a development diagnostic so numerical parity can
    /// be established before relying on final phoneme strings.
    ///
    /// # Errors
    ///
    /// Returns an error when the word contains unsupported input characters.
    pub fn trace_word(&self, word: &str) -> eyre::Result<FrontendTrace> {
        let mut output = self.embedding.forward(self.input_for_word(word)?);
        let embedding = first_position(&output)?;
        output = self.pos_encoder.forward(output);
        let positional = first_position(&output)?;
        let mut layers = Vec::with_capacity(self.encoder.len());
        let mut layer0_query = Vec::new();
        let mut layer0_key = Vec::new();
        let mut layer0_value = Vec::new();
        let mut layer0_weights = Vec::new();
        let mut layer0_context = Vec::new();
        let mut layer0_merged_context = Vec::new();
        let mut layer0_attention = Vec::new();
        let mut layer0_norm1 = Vec::new();
        let mut layer0_feed_forward = Vec::new();
        for (index, layer) in self.encoder.iter().enumerate() {
            let layer_input = output;
            let attention = layer.self_attn.forward(layer_input.clone());
            let norm1 = layer.norm1.forward(layer_input.clone() + attention.clone());
            let feed_forward = layer
                .linear2
                .forward(relu(layer.linear1.forward(norm1.clone())));
            let layer_output = layer.norm2.forward(norm1.clone() + feed_forward.clone());
            if index == 0 {
                let [batch_size, sequence_length, _] = layer_input.dims();
                let query = layer
                    .self_attn
                    .query
                    .forward(layer_input.clone())
                    .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
                    .swap_dims(1, 2);
                let key = layer
                    .self_attn
                    .key
                    .forward(layer_input.clone())
                    .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
                    .swap_dims(1, 2);
                let value = layer
                    .self_attn
                    .value
                    .forward(layer_input)
                    .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
                    .swap_dims(1, 2);
                let weights = softmax(
                    query
                        .clone()
                        .matmul(key.clone().transpose())
                        .div_scalar(ATTENTION_SCALE),
                    3,
                );
                let context = weights.clone().matmul(value.clone());
                layer0_query = first_attention_position(&query)?;
                layer0_key = first_attention_position(&key)?;
                layer0_value = first_attention_position(&value)?;
                layer0_weights = first_attention_position(&weights)?;
                layer0_context = first_attention_position(&context)?;
                layer0_merged_context =
                    first_position(&context.clone().swap_dims(1, 2).reshape([
                        batch_size,
                        sequence_length,
                        D_MODEL,
                    ]))?;
                layer0_attention = first_position(&attention)?;
                layer0_norm1 = first_position(&norm1)?;
                layer0_feed_forward = first_position(&feed_forward)?;
            }
            output = layer_output;
            layers.push(first_position(&output)?);
        }
        output = self.norm.forward(output);
        let norm = first_position(&output)?;
        let logits = first_position(&self.fc_out.forward(output))?;
        Ok(FrontendTrace {
            embedding,
            positional,
            layer0_query,
            layer0_key,
            layer0_value,
            layer0_weights,
            layer0_context,
            layer0_merged_context,
            layer0_attention,
            layer0_norm1,
            layer0_feed_forward,
            layers,
            norm,
            logits,
        })
    }

    fn input_for_word(&self, word: &str) -> eyre::Result<Tensor<B, 2, Int>> {
        let device = self.embedding.weight.val().device();
        let mut values = vec![2_i32]; // <en_us>
        for character in word.chars() {
            let Some(index) = text_symbol_index(character) else {
                bail!(
                    "unknown word {:?} contains unsupported character {:?}",
                    word,
                    character
                );
            };
            values.extend([index, index, index]);
        }
        values.push(3); // <end>
        Ok(Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(values.clone(), [1, values.len()]),
            &device,
        ))
    }

    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut output = self.embedding.forward(tokens);
        output = self.pos_encoder.forward(output);
        for layer in &self.encoder {
            output = layer.forward(output);
        }
        output = self.norm.forward(output);
        self.fc_out.forward(output)
    }
}

fn first_position<B: Backend>(tensor: &Tensor<B, 3>) -> eyre::Result<Vec<f32>> {
    let [_, _, features] = tensor.dims();
    tensor
        .clone()
        .slice([0..1, 0..1, 0..features])
        .reshape([features])
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to read frontend trace tensor: {error:?}"))
}

fn first_attention_position<B: Backend>(tensor: &Tensor<B, 4>) -> eyre::Result<Vec<f32>> {
    let [_, _, _, features] = tensor.dims();
    tensor
        .clone()
        .slice([0..1, 0..1, 0..1, 0..features])
        .reshape([features])
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("failed to read frontend attention trace tensor: {error:?}"))
}

/// Load a converter-produced phonemizer state dictionary.
///
/// # Errors
///
/// Returns an error if the state dictionary does not exactly populate the
/// native architecture.
pub fn load_glados_phonemizer_pytorch<B: Backend>(
    module: &mut GladosPhonemizer<B>,
    path: &Path,
) -> eyre::Result<()> {
    let mut store = PytorchStore::from_file(path).allow_partial(false);
    let result = module.load_from(&mut store).wrap_err_with(|| {
        format!(
            "failed to import phonemizer state dictionary {}",
            path.display()
        )
    })?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "phonemizer state dictionary {} did not match the native model (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

/// Load a prepared Burnpack containing the native phonemizer.
///
/// # Errors
///
/// Returns an error if the Burnpack does not exactly populate the model.
pub fn load_glados_phonemizer_burnpack<B: Backend>(
    module: &mut GladosPhonemizer<B>,
    path: &Path,
) -> eyre::Result<()> {
    let mut store = BurnpackStore::from_file(path).allow_partial(false);
    let result = module
        .load_from(&mut store)
        .wrap_err_with(|| format!("failed to load phonemizer Burnpack {}", path.display()))?;
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        bail!(
            "phonemizer Burnpack {} did not match the native model (errors: {:?}, missing: {:?}, unused: {:?})",
            path.display(),
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

fn text_symbol_index(character: char) -> Option<i32> {
    // 0 pad, 1 <de>, 2 <en_us>, 3 <end>, then the configured text symbols.
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß'"
        .chars()
        .position(|symbol| symbol == character)
        .and_then(|index| i32::try_from(index + 4).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_constants_match_checkpoint_contract() {
        assert_eq!(DEEP_PHONEMIZER_SYMBOLS.chars().count(), 49);
        assert_eq!(text_symbol_index('a'), Some(4));
        assert_eq!(text_symbol_index('\''), Some(63));
    }

    #[test]
    fn phonemizer_architecture_has_expected_output_shape() {
        type Backend = burn::backend::NdArray;
        let device = <Backend as burn::tensor::backend::Backend>::Device::default();
        let model = GladosPhonemizer::init(&device);
        let tokens = Tensor::<Backend, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![2_i32, 4, 4, 4, 3], [1, 5]),
            &device,
        );
        let logits = model.forward(tokens);
        assert_eq!(logits.dims(), [1, 5, 53]);
    }
}
