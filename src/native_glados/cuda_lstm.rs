//! CUDA `CubeCL` extension for the acoustic bidirectional LSTM.
//!
//! The regular Burn implementation expresses the recurrent layer as a loop of
//! tensor operations. That is portable, but it leaves a matrix-multiply and a
//! collection of gate elementwise kernels per timestep. This module keeps the
//! recurrent dependency inside one `CubeCL` launch per direction. It is
//! intentionally specialized to the batch-one, f32 CUDA inference workload;
//! callers retain the packed Burn implementation as the fallback.

use burn::nn::lstm::{BiLstm, Lstm};
use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorPrimitive;
use burn_cubecl::kernel::into_contiguous_aligned;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::barrier::{Barrier, BarrierLevel};
use cubecl::prelude::*;

type CudaRuntime = cubecl::cuda::CudaRuntime;
type CudaTensor = CubeTensor<CudaRuntime>;
type CudaBackend = burn::backend::Cuda;

const LINE_SIZE: u8 = 1;

#[derive(CubeLaunch, CubeType)]
struct LstmWeights<F: CubePrimitive> {
    input_gate: Tensor<Line<F>>,
    forget_gate: Tensor<Line<F>>,
    output_gate: Tensor<Line<F>>,
    cell_gate: Tensor<Line<F>>,
    input_hidden: Tensor<Line<F>>,
    forget_hidden: Tensor<Line<F>>,
    output_hidden: Tensor<Line<F>>,
    cell_hidden: Tensor<Line<F>>,
}

#[derive(CubeLaunch, CubeType)]
struct LstmBiases<F: CubePrimitive> {
    input_gate: Tensor<Line<F>>,
    forget_gate: Tensor<Line<F>>,
    output_gate: Tensor<Line<F>>,
    cell_gate: Tensor<Line<F>>,
    input_hidden: Tensor<Line<F>>,
    forget_hidden: Tensor<Line<F>>,
    output_hidden: Tensor<Line<F>>,
    cell_hidden: Tensor<Line<F>>,
}

/// Run the two directions through one recurrent `CubeCL` kernel each.
pub(super) fn forward_bidirectional_lstm(
    lstm: &BiLstm<CudaBackend>,
    input: BurnTensor<CudaBackend, 3>,
) -> BurnTensor<CudaBackend, 3> {
    let forward = forward_lstm_direction(&lstm.forward, input.clone());
    let reverse = forward_lstm_direction(&lstm.reverse, input.flip([1])).flip([1]);
    BurnTensor::cat(vec![forward, reverse], 2)
}

#[expect(
    clippy::too_many_lines,
    reason = "The host launch keeps the specialized LSTM parameter binding and fallback together."
)]
fn forward_lstm_direction(
    lstm: &Lstm<CudaBackend>,
    input: BurnTensor<CudaBackend, 3>,
) -> BurnTensor<CudaBackend, 3> {
    let [batch_size, sequence_length, input_channels] = input.dims();
    let hidden_channels = lstm.d_hidden;

    assert_eq!(
        batch_size, 1,
        "CUDA fused LSTM only supports batch size one"
    );
    assert!(
        input_channels > 0,
        "CUDA fused LSTM requires input channels"
    );
    assert!(
        hidden_channels > 0,
        "CUDA fused LSTM requires hidden channels"
    );

    let input = into_contiguous_aligned(float_primitive(input));
    let input_gate_weight = parameter_tensor(&lstm.input_gate.input_transform.weight);
    let forget_gate_weight = parameter_tensor(&lstm.forget_gate.input_transform.weight);
    let output_gate_weight = parameter_tensor(&lstm.output_gate.input_transform.weight);
    let cell_gate_weight = parameter_tensor(&lstm.cell_gate.input_transform.weight);
    let input_hidden_weight = parameter_tensor(&lstm.input_gate.hidden_transform.weight);
    let forget_hidden_weight = parameter_tensor(&lstm.forget_gate.hidden_transform.weight);
    let output_hidden_weight = parameter_tensor(&lstm.output_gate.hidden_transform.weight);
    let cell_hidden_weight = parameter_tensor(&lstm.cell_gate.hidden_transform.weight);

    let input_for_fallback = || BurnTensor::from_primitive(TensorPrimitive::Float(input.clone()));
    let Some(input_gate_bias) = parameter_bias(lstm.input_gate.input_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(forget_gate_bias) = parameter_bias(lstm.forget_gate.input_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(output_gate_bias) = parameter_bias(lstm.output_gate.input_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(cell_gate_bias) = parameter_bias(lstm.cell_gate.input_transform.bias.as_ref()) else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(input_hidden_bias) = parameter_bias(lstm.input_gate.hidden_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(forget_hidden_bias) = parameter_bias(lstm.forget_gate.hidden_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(output_hidden_bias) = parameter_bias(lstm.output_gate.hidden_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };
    let Some(cell_hidden_bias) = parameter_bias(lstm.cell_gate.hidden_transform.bias.as_ref())
    else {
        return super::forward_lstm_direction_reference(lstm, input_for_fallback());
    };

    let device = input.device.clone();
    let client = input.client.clone();
    let output_allocation = client.empty_tensor(
        &[batch_size, sequence_length, hidden_channels],
        size_of::<f32>(),
    );
    let output = CubeTensor::new(
        client.clone(),
        output_allocation.handle,
        [batch_size, sequence_length, hidden_channels].into(),
        device,
        output_allocation.strides,
        burn::tensor::DType::F32,
    );
    let hidden_channels_u32 =
        u32::try_from(hidden_channels).expect("hidden channels fit CubeCL dimensions");
    let input_channels_u32 =
        u32::try_from(input_channels).expect("input channels fit CubeCL dimensions");

    // SAFETY: the launch uses one cube with `hidden_channels` units, matching
    // the shared-state indexing and the output allocation dimensions. All
    // tensor arguments are contiguous/aligned f32 CUDA allocations, and the
    // compile-time channel counts match the model parameter shapes.
    unsafe {
        lstm_direction::launch_unchecked::<f32, CudaRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(hidden_channels_u32, 1, 1),
            input.as_tensor_arg::<f32>(LINE_SIZE),
            LstmWeightsLaunch::new(
                input_gate_weight.as_tensor_arg::<f32>(LINE_SIZE),
                forget_gate_weight.as_tensor_arg::<f32>(LINE_SIZE),
                output_gate_weight.as_tensor_arg::<f32>(LINE_SIZE),
                cell_gate_weight.as_tensor_arg::<f32>(LINE_SIZE),
                input_hidden_weight.as_tensor_arg::<f32>(LINE_SIZE),
                forget_hidden_weight.as_tensor_arg::<f32>(LINE_SIZE),
                output_hidden_weight.as_tensor_arg::<f32>(LINE_SIZE),
                cell_hidden_weight.as_tensor_arg::<f32>(LINE_SIZE),
            ),
            LstmBiasesLaunch::new(
                input_gate_bias.as_tensor_arg::<f32>(LINE_SIZE),
                forget_gate_bias.as_tensor_arg::<f32>(LINE_SIZE),
                output_gate_bias.as_tensor_arg::<f32>(LINE_SIZE),
                cell_gate_bias.as_tensor_arg::<f32>(LINE_SIZE),
                input_hidden_bias.as_tensor_arg::<f32>(LINE_SIZE),
                forget_hidden_bias.as_tensor_arg::<f32>(LINE_SIZE),
                output_hidden_bias.as_tensor_arg::<f32>(LINE_SIZE),
                cell_hidden_bias.as_tensor_arg::<f32>(LINE_SIZE),
            ),
            output.as_tensor_arg::<f32>(LINE_SIZE),
            input_channels_u32,
            hidden_channels_u32,
        );
    };

    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

fn float_primitive<const D: usize>(tensor: BurnTensor<CudaBackend, D>) -> CudaTensor {
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        TensorPrimitive::QFloat(_) => panic!("CUDA fused LSTM received a quantized tensor"),
    }
}

fn parameter_tensor(parameter: &burn::module::Param<BurnTensor<CudaBackend, 2>>) -> CudaTensor {
    into_contiguous_aligned(float_primitive(parameter.val()))
}

fn parameter_bias(
    parameter: Option<&burn::module::Param<BurnTensor<CudaBackend, 1>>>,
) -> Option<CudaTensor> {
    parameter.map(|parameter| into_contiguous_aligned(float_primitive(parameter.val())))
}

#[expect(
    clippy::needless_pass_by_value,
    clippy::trivially_copy_pass_by_ref,
    reason = "CubeCL launch arguments are handle-like frontend values and this signature is part of the generated kernel ABI."
)]
#[cube(launch_unchecked)]
fn lstm_direction<F: Float>(
    input: &Tensor<Line<F>>,
    weights: LstmWeights<F>,
    biases: LstmBiases<F>,
    output: &mut Tensor<Line<F>>,
    #[comptime] input_channels: u32,
    #[comptime] hidden_channels: u32,
) {
    let input_gate_weight = weights.input_gate;
    let forget_gate_weight = weights.forget_gate;
    let output_gate_weight = weights.output_gate;
    let cell_gate_weight = weights.cell_gate;
    let input_hidden_weight = weights.input_hidden;
    let forget_hidden_weight = weights.forget_hidden;
    let output_hidden_weight = weights.output_hidden;
    let cell_hidden_weight = weights.cell_hidden;
    let input_gate_bias = biases.input_gate;
    let forget_gate_bias = biases.forget_gate;
    let output_gate_bias = biases.output_gate;
    let cell_gate_bias = biases.cell_gate;
    let input_hidden_bias = biases.input_hidden;
    let forget_hidden_bias = biases.forget_hidden;
    let output_hidden_bias = biases.output_hidden;
    let cell_hidden_bias = biases.cell_hidden;
    let cell_state = SharedMemory::<F>::new_lined(hidden_channels, 1u32);
    let hidden_state = SharedMemory::<F>::new_lined(hidden_channels, 1u32);
    let barrier = Barrier::new(BarrierLevel::cube_manual(0u32));
    sync_cube();

    let hidden_index = UNIT_POS_X;
    if hidden_index < hidden_channels {
        cell_state.write(hidden_index, Line::new(F::new(0.0)));
        hidden_state.write(hidden_index, Line::new(F::new(0.0)));
    }
    sync_cube();

    for time_index in 0..input.shape(1) {
        let mut input_gate_projection = Line::new(F::new(0.0));
        let mut forget_gate_projection = Line::new(F::new(0.0));
        let mut output_gate_projection = Line::new(F::new(0.0));
        let mut cell_gate_projection = Line::new(F::new(0.0));

        for input_index in 0..input_channels {
            let input_index_linear = time_index * input.stride(1) + input_index * input.stride(2);
            let input_value = input.read(input_index_linear);
            let input_weight_index = input_index * input_gate_weight.stride(0)
                + hidden_index * input_gate_weight.stride(1);
            let forget_weight_index = input_index * forget_gate_weight.stride(0)
                + hidden_index * forget_gate_weight.stride(1);
            let output_weight_index = input_index * output_gate_weight.stride(0)
                + hidden_index * output_gate_weight.stride(1);
            let cell_weight_index = input_index * cell_gate_weight.stride(0)
                + hidden_index * cell_gate_weight.stride(1);
            input_gate_projection += input_value * input_gate_weight.read(input_weight_index);
            forget_gate_projection += input_value * forget_gate_weight.read(forget_weight_index);
            output_gate_projection += input_value * output_gate_weight.read(output_weight_index);
            cell_gate_projection += input_value * cell_gate_weight.read(cell_weight_index);
        }

        for hidden_source in 0..hidden_channels {
            let hidden_value = hidden_state.read(hidden_source);
            let input_hidden_index = hidden_source * input_hidden_weight.stride(0)
                + hidden_index * input_hidden_weight.stride(1);
            let forget_hidden_index = hidden_source * forget_hidden_weight.stride(0)
                + hidden_index * forget_hidden_weight.stride(1);
            let output_hidden_index = hidden_source * output_hidden_weight.stride(0)
                + hidden_index * output_hidden_weight.stride(1);
            let cell_hidden_index = hidden_source * cell_hidden_weight.stride(0)
                + hidden_index * cell_hidden_weight.stride(1);
            input_gate_projection += hidden_value * input_hidden_weight.read(input_hidden_index);
            forget_gate_projection += hidden_value * forget_hidden_weight.read(forget_hidden_index);
            output_gate_projection += hidden_value * output_hidden_weight.read(output_hidden_index);
            cell_gate_projection += hidden_value * cell_hidden_weight.read(cell_hidden_index);
        }

        let input_gate = sigmoid(
            input_gate_projection
                + input_gate_bias.read(hidden_index)
                + input_hidden_bias.read(hidden_index),
        );
        let forget_gate = sigmoid(
            forget_gate_projection
                + forget_gate_bias.read(hidden_index)
                + forget_hidden_bias.read(hidden_index),
        );
        let output_gate = sigmoid(
            output_gate_projection
                + output_gate_bias.read(hidden_index)
                + output_hidden_bias.read(hidden_index),
        );
        let candidate = Line::tanh(
            cell_gate_projection
                + cell_gate_bias.read(hidden_index)
                + cell_hidden_bias.read(hidden_index),
        );

        let next_cell = forget_gate * cell_state.read(hidden_index) + input_gate * candidate;
        let next_hidden = output_gate * Line::tanh(next_cell);
        cell_state.write(hidden_index, next_cell);
        hidden_state.write(hidden_index, next_hidden);
        output.write(
            time_index * output.stride(1) + hidden_index * output.stride(2),
            next_hidden,
        );

        barrier.arrive_and_wait();
    }
}

#[cube]
fn sigmoid<F: Float>(value: Line<F>) -> Line<F> {
    Line::new(F::new(1.0)) / (Line::new(F::new(1.0)) + Line::exp(-value))
}
