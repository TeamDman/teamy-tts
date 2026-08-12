//! Burn-tch recurrent specialization backed by one LibTorch GRU operation.
//!
//! Burn's portable GRU implementation exposes the three gate projections as
//! separate tensor operations. LibTorch can execute both directions through
//! the cuDNN-backed GRU primitive instead.

use super::BidirectionalGru;
use burn::backend::LibTorch;
use burn::backend::libtorch::TchTensor;
use burn::nn::gru::Gru;
use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;
use tch::Tensor as TorchTensor;

type TchBackend = LibTorch<f32>;

/// Run both predictor GRU directions through one LibTorch bidirectional op.
pub(super) fn forward_bidirectional_gru(
    gru: &BidirectionalGru<TchBackend>,
    input: Tensor<TchBackend, 3>,
) -> Tensor<TchBackend, 3> {
    let input = float_tensor(input);
    let (batch_size, _sequence_length, _input_channels) = input.size3().unwrap();
    let hidden_channels =
        i64::try_from(gru.forward.d_hidden).expect("LibTorch GRU hidden channels fit i64");
    let hidden = TorchTensor::zeros(
        [2, batch_size, hidden_channels],
        (tch::Kind::Float, input.device()),
    );

    let mut parameters = Vec::with_capacity(8);
    append_direction_parameters(&mut parameters, &gru.forward);
    append_direction_parameters(&mut parameters, &gru.reverse);
    let (output, _) = input.gru(&hidden, &parameters, true, 1, 0.0, false, true, true);

    Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(output)))
}

fn append_direction_parameters(parameters: &mut Vec<TorchTensor>, gru: &Gru<TchBackend>) {
    // PyTorch's GRU parameter order is reset, update, new. Burn stores the
    // same gates as update, reset, new to match its public module layout.
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
        TensorPrimitive::QFloat(_) => panic!("LibTorch GRU received a quantized tensor"),
    }
}

fn float_tensor(tensor: Tensor<TchBackend, 3>) -> TorchTensor {
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch GRU received a quantized tensor"),
    }
}
