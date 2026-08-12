//! Burn-tch implementation of the CBHG block using LibTorch tensor kernels.

use super::BidirectionalGru;
use super::Cbhg;
use super::ConvNormBlock;
use super::ConvNormBlockNoRelu;
use super::Highway;
use burn::backend::LibTorch;
use burn::backend::libtorch::TchTensor;
use burn::nn::BatchNorm;
use burn::nn::Linear;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;
use tch::Tensor as TorchTensor;

type TchBackend = LibTorch<f32>;

/// Execute CBHG without routing each convolution, normalization, and highway
/// operation through Burn's per-operation backend boundary.
pub(super) fn forward_cbhg(
    cbhg: &Cbhg<TchBackend>,
    input: Tensor<TchBackend, 3>,
) -> Tensor<TchBackend, 3> {
    let input = float_tensor(input);
    let sequence_length = input.size3().unwrap().2;

    let bank = cbhg
        .conv1d_bank
        .iter()
        .map(|conv| conv_norm_relu(&input, conv).slice(2, 0, sequence_length, 1))
        .collect::<Vec<_>>();
    let bank = TorchTensor::cat(&bank, 1);
    let pooled = bank
        .max_pool1d([2], [1], [1], [1], false)
        .slice(2, 0, sequence_length, 1);
    let projected = conv_norm_relu(&pooled, &cbhg.conv_project1);
    let projected = conv_norm(&projected, &cbhg.conv_project2);
    let mut output = linear(&(projected + input).transpose(1, 2), &cbhg.pre_highway);
    for highway in &cbhg.highways {
        output = highway_forward(&output, highway);
    }
    let (output, _) = gru_forward(&output, &cbhg.rnn);

    Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(output)))
}

pub(super) fn conv_norm_relu(
    input: &TorchTensor,
    block: &ConvNormBlock<TchBackend>,
) -> TorchTensor {
    let output = conv1d(input, &block.conv).relu();
    batch_norm(&output, &block.bnorm)
}

fn conv_norm(input: &TorchTensor, block: &ConvNormBlockNoRelu<TchBackend>) -> TorchTensor {
    batch_norm(&conv1d(input, &block.conv), &block.bnorm)
}

fn conv1d(input: &TorchTensor, conv: &Conv1d<TchBackend>) -> TorchTensor {
    let padding = padding_value(&conv.padding.0, conv.kernel_size);
    let weight = tensor_ref(&conv.weight.val());
    let bias = conv.bias.as_ref().map(|bias| tensor_ref(&bias.val()));
    input.conv1d(
        &weight,
        bias.as_ref(),
        [i64::try_from(conv.stride).expect("convolution stride fits i64")],
        [padding],
        [i64::try_from(conv.dilation).expect("convolution dilation fits i64")],
        i64::try_from(conv.groups).expect("convolution groups fit i64"),
    )
}

fn padding_value(padding: &PaddingConfig1d, kernel_size: usize) -> i64 {
    match padding {
        PaddingConfig1d::Valid => 0,
        PaddingConfig1d::Explicit(value) => i64::try_from(*value).expect("padding fits i64"),
        PaddingConfig1d::Same => i64::try_from(kernel_size / 2).expect("padding fits i64"),
    }
}

fn batch_norm(input: &TorchTensor, norm: &BatchNorm<TchBackend>) -> TorchTensor {
    let gamma = tensor_ref(&norm.gamma.val());
    let beta = tensor_ref(&norm.beta.val());
    let mean = tensor_ref(&norm.running_mean.value());
    let variance = tensor_ref(&norm.running_var.value());
    input.batch_norm(
        Some(&gamma),
        Some(&beta),
        Some(&mean),
        Some(&variance),
        false,
        norm.momentum,
        norm.epsilon,
        true,
    )
}

pub(super) fn linear(input: &TorchTensor, layer: &Linear<TchBackend>) -> TorchTensor {
    let weight = tensor_ref(&layer.weight.val());
    let mut output = input.matmul(&weight);
    if let Some(bias) = &layer.bias {
        output += tensor_ref(&bias.val());
    }
    output
}

fn highway_forward(input: &TorchTensor, highway: &Highway<TchBackend>) -> TorchTensor {
    let gate = linear(input, &highway.w2).sigmoid();
    let transformed = linear(input, &highway.w1).relu();
    &gate * &transformed + (gate.ones_like() - gate) * input
}

pub(super) fn gru_forward(
    input: &TorchTensor,
    gru: &BidirectionalGru<TchBackend>,
) -> (TorchTensor, TorchTensor) {
    let batch_size = input.size3().unwrap().0;
    let hidden_channels =
        i64::try_from(gru.forward.d_hidden).expect("LibTorch GRU hidden channels fit i64");
    let hidden = TorchTensor::zeros(
        [2, batch_size, hidden_channels],
        (tch::Kind::Float, input.device()),
    );
    let mut parameters = Vec::with_capacity(8);
    append_gru_parameters(&mut parameters, &gru.forward);
    append_gru_parameters(&mut parameters, &gru.reverse);
    input.gru(&hidden, &parameters, true, 1, 0.0, false, true, true)
}

fn append_gru_parameters(parameters: &mut Vec<TorchTensor>, gru: &burn::nn::gru::Gru<TchBackend>) {
    parameters.push(gate_weights(
        &gru.reset_gate.input_transform.weight.val(),
        &gru.update_gate.input_transform.weight.val(),
        &gru.new_gate.input_transform.weight.val(),
    ));
    parameters.push(gate_weights(
        &gru.reset_gate.hidden_transform.weight.val(),
        &gru.update_gate.hidden_transform.weight.val(),
        &gru.new_gate.hidden_transform.weight.val(),
    ));
    parameters.push(gate_biases(
        &gru.reset_gate.input_transform.bias,
        &gru.update_gate.input_transform.bias,
        &gru.new_gate.input_transform.bias,
    ));
    parameters.push(gate_biases(
        &gru.reset_gate.hidden_transform.bias,
        &gru.update_gate.hidden_transform.bias,
        &gru.new_gate.hidden_transform.bias,
    ));
}

fn gate_weights(
    reset: &Tensor<TchBackend, 2>,
    update: &Tensor<TchBackend, 2>,
    new: &Tensor<TchBackend, 2>,
) -> TorchTensor {
    TorchTensor::cat(&[tensor_ref(reset), tensor_ref(update), tensor_ref(new)], 1)
        .transpose(0, 1)
        .contiguous()
}

fn gate_biases(
    reset: &Option<burn::module::Param<Tensor<TchBackend, 1>>>,
    update: &Option<burn::module::Param<Tensor<TchBackend, 1>>>,
    new: &Option<burn::module::Param<Tensor<TchBackend, 1>>>,
) -> TorchTensor {
    let reset = reset.as_ref().expect("GLaDOS GRU requires bias tensors");
    let update = update.as_ref().expect("GLaDOS GRU requires bias tensors");
    let new = new.as_ref().expect("GLaDOS GRU requires bias tensors");
    TorchTensor::cat(
        &[
            tensor_ref(&reset.val()),
            tensor_ref(&update.val()),
            tensor_ref(&new.val()),
        ],
        0,
    )
}

fn tensor_ref<const D: usize>(tensor: &Tensor<TchBackend, D>) -> TorchTensor {
    match tensor.clone().into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch CBHG received a quantized tensor"),
    }
}

pub(super) fn float_tensor(tensor: Tensor<TchBackend, 3>) -> TorchTensor {
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch CBHG received a quantized tensor"),
    }
}
