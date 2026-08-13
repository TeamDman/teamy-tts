//! Direct `tch` implementation of the upstream `DeepPhonemizer` forward model.
//!
//! The checkpoint is exported by `tools/export_glados_phonemizer.py` as a
//! named `PyTorch` tensor archive. Keeping the model in `LibTorch` tensors avoids
//! a Burnpack dependency while preserving the exact native architecture.

use eyre::Context;
use eyre::bail;
use std::path::Path;
use tch::Device;
use tch::Kind;
use tch::Tensor;
use tch::nn;

const D_MODEL: i64 = 512;
const D_FF: i64 = 1024;
const HEADS: i64 = 4;
const HEAD_DIM: i64 = D_MODEL / HEADS;
const ATTENTION_SCALE: f64 = 11.313_708;
const LAYERS: usize = 6;
const TEXT_VOCAB: i64 = 64;
const PHONEME_VOCAB: i64 = 53;
const MAX_SEQUENCE: i64 = 5000;

/// IPA symbols emitted by the upstream checkpoint.
pub const DEEP_PHONEMIZER_SYMBOLS: &str = "abdefghijklmnoprstuvwxyzæçðøŋœɐɑɔəɛɝɹɡɪʁʃʊʌʏʒʔː͡θ";

#[derive(Debug)]
struct LinearWeights {
    weight: Tensor,
    bias: Tensor,
}

impl LinearWeights {
    fn new(root: &nn::Path<'_>, prefix: &str, input: i64, output: i64) -> Self {
        Self {
            weight: named_var(
                root,
                &format!("{prefix}.weight"),
                &[output, input],
                nn::Init::Const(0.),
            ),
            bias: named_var(
                root,
                &format!("{prefix}.bias"),
                &[output],
                nn::Init::Const(0.),
            ),
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.linear(&self.weight, Some(&self.bias))
    }
}

#[derive(Debug)]
struct LayerNormWeights {
    gamma: Tensor,
    beta: Tensor,
}

impl LayerNormWeights {
    fn new(root: &nn::Path<'_>, prefix: &str) -> Self {
        Self {
            gamma: named_var(
                root,
                &format!("{prefix}.gamma"),
                &[D_MODEL],
                nn::Init::Const(1.),
            ),
            beta: named_var(
                root,
                &format!("{prefix}.beta"),
                &[D_MODEL],
                nn::Init::Const(0.),
            ),
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.layer_norm([D_MODEL], Some(&self.gamma), Some(&self.beta), 1e-5, false)
    }
}

#[derive(Debug)]
struct AttentionWeights {
    query: LinearWeights,
    key: LinearWeights,
    value: LinearWeights,
    out_proj: LinearWeights,
}

impl AttentionWeights {
    fn new(root: &nn::Path<'_>, prefix: &str) -> Self {
        Self {
            query: LinearWeights::new(root, &format!("{prefix}.query"), D_MODEL, D_MODEL),
            key: LinearWeights::new(root, &format!("{prefix}.key"), D_MODEL, D_MODEL),
            value: LinearWeights::new(root, &format!("{prefix}.value"), D_MODEL, D_MODEL),
            out_proj: LinearWeights::new(root, &format!("{prefix}.out_proj"), D_MODEL, D_MODEL),
        }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let (batch_size, sequence_length, _) = input.size3().expect("phonemizer input is rank 3");
        let query = self
            .query
            .forward(input)
            .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
            .transpose(1, 2);
        let key = self
            .key
            .forward(input)
            .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
            .transpose(1, 2);
        let value = self
            .value
            .forward(input)
            .reshape([batch_size, sequence_length, HEADS, HEAD_DIM])
            .transpose(1, 2);
        let attention = query
            .matmul(&key.transpose(-2, -1))
            .divide_scalar(ATTENTION_SCALE)
            .softmax(-1, Kind::Float);
        self.out_proj
            .forward(&attention.matmul(&value).transpose(1, 2).reshape([
                batch_size,
                sequence_length,
                D_MODEL,
            ]))
    }
}

#[derive(Debug)]
struct EncoderLayer {
    self_attn: AttentionWeights,
    linear1: LinearWeights,
    linear2: LinearWeights,
    norm1: LayerNormWeights,
    norm2: LayerNormWeights,
}

impl EncoderLayer {
    fn new(root: &nn::Path<'_>, index: usize) -> Self {
        let prefix = format!("encoder.{index}");
        Self {
            self_attn: AttentionWeights::new(root, &format!("{prefix}.self_attn")),
            linear1: LinearWeights::new(root, &format!("{prefix}.linear1"), D_MODEL, D_FF),
            linear2: LinearWeights::new(root, &format!("{prefix}.linear2"), D_FF, D_MODEL),
            norm1: LayerNormWeights::new(root, &format!("{prefix}.norm1")),
            norm2: LayerNormWeights::new(root, &format!("{prefix}.norm2")),
        }
    }

    fn forward(&self, input: Tensor) -> Tensor {
        let attention = self.self_attn.forward(&input);
        let input = self.norm1.forward(&(input + attention));
        let feed_forward = self.linear2.forward(&self.linear1.forward(&input).relu());
        self.norm2.forward(&(input + feed_forward))
    }
}

/// The native `LibTorch` `DeepPhonemizer` model used for dictionary misses.
#[derive(Debug)]
pub struct GladosPhonemizer {
    variables: nn::VarStore,
    embedding: Tensor,
    positional_scale: Tensor,
    positional_encoding: Tensor,
    encoder: Vec<EncoderLayer>,
    norm: LayerNormWeights,
    fc_out: LinearWeights,
}

impl GladosPhonemizer {
    /// Load the exported named tensor archive on the CPU.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor archive cannot be loaded into the
    /// expected named model variables.
    pub fn from_file(path: &Path) -> eyre::Result<Self> {
        let variables = nn::VarStore::new(Device::Cpu);
        let root = variables.root();
        let embedding = named_var(
            &root,
            "embedding.weight",
            &[TEXT_VOCAB, D_MODEL],
            nn::Init::Const(0.),
        );
        let positional_scale = named_var(&root, "pos_encoder.scale", &[1], nn::Init::Const(1.));
        let positional_encoding = named_var(
            &root,
            "pos_encoder.pe",
            &[MAX_SEQUENCE, 1, D_MODEL],
            nn::Init::Const(0.),
        );
        let encoder = (0..LAYERS)
            .map(|index| EncoderLayer::new(&root, index))
            .collect();
        let norm = LayerNormWeights::new(&root, "norm");
        let fc_out = LinearWeights::new(&root, "fc_out", D_MODEL, PHONEME_VOCAB);
        let mut model = Self {
            variables,
            embedding,
            positional_scale,
            positional_encoding,
            encoder,
            norm,
            fc_out,
        };
        model.variables.load(path).wrap_err_with(|| {
            format!("failed to load tch phonemizer tensors {}", path.display())
        })?;
        Ok(model)
    }

    /// Predict IPA for one dictionary-missing word.
    ///
    /// # Errors
    ///
    /// Returns an error when the word contains unsupported characters or the
    /// model output cannot be converted into IPA symbols.
    pub fn phonemize_word(&self, word: &str) -> eyre::Result<String> {
        let logits = self.forward(&Self::input_for_word(word)?);
        let indices = logits.argmax(-1, false).to_device(Device::Cpu).view([-1]);
        let indices =
            Vec::<i64>::try_from(indices).wrap_err("failed to read tch phonemizer output")?;
        let mut output = String::new();
        let mut previous = None;
        for index in indices {
            if index == 3 {
                break;
            }
            if index <= 2 {
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

    fn input_for_word(word: &str) -> eyre::Result<Tensor> {
        let mut values = vec![2_i64];
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
        values.push(3);
        let sequence_length =
            i64::try_from(values.len()).wrap_err("phonemizer input sequence length exceeds i64")?;
        Ok(Tensor::from_slice(&values).reshape([1, sequence_length]))
    }

    fn forward(&self, tokens: &Tensor) -> Tensor {
        let sequence_length = tokens.size()[1];
        let mut output = Tensor::embedding(&self.embedding, tokens, -1, false, false);
        let position = self
            .positional_encoding
            .narrow(0, 0, sequence_length)
            .transpose(0, 1);
        output += position * &self.positional_scale;
        for layer in &self.encoder {
            output = layer.forward(output);
        }
        output = self.norm.forward(&output);
        self.fc_out.forward(&output)
    }
}

fn text_symbol_index(character: char) -> Option<i64> {
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß'"
        .chars()
        .position(|symbol| symbol == character)
        .and_then(|index| i64::try_from(index + 4).ok())
}

fn named_var(root: &nn::Path<'_>, name: &str, dims: &[i64], init: nn::Init) -> Tensor {
    let components = name.split('.').collect::<Vec<_>>();
    let (leaf, parents) = components
        .split_last()
        .expect("named tensor path must contain a leaf");
    let path = parents
        .iter()
        .fold(root.clone(), |path, component| path.sub(*component));
    path.var(leaf, dims, init)
}
