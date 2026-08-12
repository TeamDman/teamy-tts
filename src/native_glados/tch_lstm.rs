//! Burn-tch recurrent specialization backed by one LibTorch RNN operation.
//!
//! Burn's generic LSTM module is intentionally portable, but its operation
//! graph exposes the four gate projections separately. The tch backend can
//! hand the same Burn-owned parameters to LibTorch's fused bidirectional LSTM
//! operator, which is the backend-native path to cuDNN's persistent RNN
//! implementation on CUDA.

use burn::backend::LibTorch;
use burn::backend::libtorch::TchTensor;
use burn::nn::lstm::BiLstm;
use burn::tensor::Tensor;
use burn::tensor::TensorPrimitive;
use tch::Tensor as TorchTensor;

type TchBackend = LibTorch<f32>;

/// Run both acoustic LSTM directions through one LibTorch bidirectional op.
///
/// The Burn module stores each gate as an `[input, hidden]` matrix and keeps
/// input/hidden biases separately. LibTorch expects `[4 * hidden, input]`,
/// `[4 * hidden, hidden]`, and the gate order `input, forget, cell, output`.
pub(super) fn forward_bidirectional_lstm(
    lstm: &BiLstm<TchBackend>,
    input: Tensor<TchBackend, 3>,
) -> Tensor<TchBackend, 3> {
    let input = float_tensor(input);
    let (batch_size, _sequence_length, _input_channels) = input.size3().unwrap();
    let hidden_channels =
        i64::try_from(lstm.forward.d_hidden).expect("LibTorch LSTM hidden channels fit i64");

    let mut parameters = Vec::with_capacity(8);
    append_direction_parameters(&mut parameters, &lstm.forward);
    append_direction_parameters(&mut parameters, &lstm.reverse);

    let hx = TorchTensor::zeros(
        [2, batch_size, hidden_channels],
        (tch::Kind::Float, input.device()),
    );
    let cx = TorchTensor::zeros(
        [2, batch_size, hidden_channels],
        (tch::Kind::Float, input.device()),
    );
    let state = [hx, cx];
    let (output, _, _) = input.lstm(&state, &parameters, true, 1, 0.0, false, true, true);

    Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(output)))
}

fn append_direction_parameters(
    parameters: &mut Vec<TorchTensor>,
    lstm: &burn::nn::lstm::Lstm<TchBackend>,
) {
    parameters.push(gate_input_weights(lstm));
    parameters.push(gate_hidden_weights(lstm));
    parameters.push(gate_input_biases(lstm));
    parameters.push(gate_hidden_biases(lstm));
}

fn gate_input_weights(lstm: &burn::nn::lstm::Lstm<TchBackend>) -> TorchTensor {
    TorchTensor::cat(
        &[
            tensor_ref(&lstm.input_gate.input_transform.weight.val()),
            tensor_ref(&lstm.forget_gate.input_transform.weight.val()),
            tensor_ref(&lstm.cell_gate.input_transform.weight.val()),
            tensor_ref(&lstm.output_gate.input_transform.weight.val()),
        ],
        1,
    )
    .transpose(0, 1)
    .contiguous()
}

fn gate_hidden_weights(lstm: &burn::nn::lstm::Lstm<TchBackend>) -> TorchTensor {
    TorchTensor::cat(
        &[
            tensor_ref(&lstm.input_gate.hidden_transform.weight.val()),
            tensor_ref(&lstm.forget_gate.hidden_transform.weight.val()),
            tensor_ref(&lstm.cell_gate.hidden_transform.weight.val()),
            tensor_ref(&lstm.output_gate.hidden_transform.weight.val()),
        ],
        1,
    )
    .transpose(0, 1)
    .contiguous()
}

fn gate_input_biases(lstm: &burn::nn::lstm::Lstm<TchBackend>) -> TorchTensor {
    TorchTensor::cat(
        &[
            bias_ref(&lstm.input_gate.input_transform.bias),
            bias_ref(&lstm.forget_gate.input_transform.bias),
            bias_ref(&lstm.cell_gate.input_transform.bias),
            bias_ref(&lstm.output_gate.input_transform.bias),
        ],
        0,
    )
}

fn gate_hidden_biases(lstm: &burn::nn::lstm::Lstm<TchBackend>) -> TorchTensor {
    TorchTensor::cat(
        &[
            bias_ref(&lstm.input_gate.hidden_transform.bias),
            bias_ref(&lstm.forget_gate.hidden_transform.bias),
            bias_ref(&lstm.cell_gate.hidden_transform.bias),
            bias_ref(&lstm.output_gate.hidden_transform.bias),
        ],
        0,
    )
}

fn tensor_ref<const D: usize>(tensor: &Tensor<TchBackend, D>) -> TorchTensor {
    match tensor.clone().into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch LSTM received a quantized tensor"),
    }
}

fn bias_ref(bias: &Option<burn::module::Param<Tensor<TchBackend, 1>>>) -> TorchTensor {
    let bias = bias.as_ref().expect("GLaDOS LSTM requires bias tensors");
    tensor_ref(&bias.val())
}

fn float_tensor(tensor: Tensor<TchBackend, 3>) -> TorchTensor {
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor.tensor,
        TensorPrimitive::QFloat(_) => panic!("LibTorch LSTM received a quantized tensor"),
    }
}
