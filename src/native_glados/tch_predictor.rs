//! Burn-tch predictor specialization using LibTorch convolution and GRU ops.

use super::ConditionalSeriesPredictor;
use super::SeriesPredictor;
use super::tch_cbhg;
use burn::backend::LibTorch;
use burn::backend::libtorch::TchTensor;
use burn::nn::Embedding;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;
use tch::Tensor as TorchTensor;

type TchBackend = LibTorch<f32>;

pub(super) fn forward_series_predictor(
    predictor: &SeriesPredictor<TchBackend>,
    tokens: Tensor<TchBackend, 2, Int>,
    speaker_embedding: Tensor<TchBackend, 2>,
    alpha: f32,
) -> Tensor<TchBackend, 3> {
    let tokens = tokens.into_primitive().tensor;
    let speaker_embedding = float_tensor(speaker_embedding);
    let (_, sequence_length) = tokens.size2().unwrap();
    let embedded = embedding(&predictor.embedding, &tokens);
    let speaker_embedding = speaker_embedding
        .unsqueeze(1)
        .repeat([1, sequence_length, 1]);
    let mut input = TorchTensor::cat(&[embedded, speaker_embedding], 2).transpose(1, 2);
    for conv in &predictor.convs {
        input = tch_cbhg::conv_norm_relu(&input, conv);
    }
    let (output, _) = tch_cbhg::gru_forward(&input.transpose(1, 2), &predictor.rnn);
    let output = tch_cbhg::linear(&output, &predictor.lin) / f64::from(alpha);
    wrap(output)
}

pub(super) fn forward_conditional_series_predictor(
    predictor: &ConditionalSeriesPredictor<TchBackend>,
    tokens: Tensor<TchBackend, 2, Int>,
    pitch_conditions: Tensor<TchBackend, 2, Int>,
    speaker_embedding: Tensor<TchBackend, 2>,
    alpha: f32,
) -> Tensor<TchBackend, 3> {
    let tokens = tokens.into_primitive().tensor;
    let pitch_conditions = pitch_conditions.into_primitive().tensor;
    let speaker_embedding = float_tensor(speaker_embedding);
    let (_, sequence_length) = tokens.size2().unwrap();
    let embedded = embedding(&predictor.embedding, &tokens);
    let pitch_conditions = embedding(&predictor.pitch_cond_embedding, &pitch_conditions);
    let speaker_embedding = speaker_embedding
        .unsqueeze(1)
        .repeat([1, sequence_length, 1]);
    let mut input =
        TorchTensor::cat(&[embedded, pitch_conditions, speaker_embedding], 2).transpose(1, 2);
    for conv in &predictor.convs {
        input = tch_cbhg::conv_norm_relu(&input, conv);
    }
    let (output, _) = tch_cbhg::gru_forward(&input.transpose(1, 2), &predictor.rnn);
    let output = tch_cbhg::linear(&output, &predictor.lin) / f64::from(alpha);
    wrap(output)
}

fn embedding(embedding: &Embedding<TchBackend>, indices: &TorchTensor) -> TorchTensor {
    let (batch_size, sequence_length) = indices.size2().unwrap();
    let [_, embedding_dimension] = embedding.weight.val().dims();
    let weight = tensor_ref(&embedding.weight.val());
    weight.index_select(0, &indices.reshape([-1])).reshape([
        batch_size,
        sequence_length,
        embedding_dimension as i64,
    ])
}

fn float_tensor(tensor: Tensor<TchBackend, 2>) -> TorchTensor {
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch predictor received a quantized tensor"),
    }
}

fn tensor_ref<const D: usize>(tensor: &Tensor<TchBackend, D>) -> TorchTensor {
    match tensor.clone().into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch predictor received a quantized tensor"),
    }
}

fn wrap(tensor: TorchTensor) -> Tensor<TchBackend, 3> {
    Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(tensor)))
}
