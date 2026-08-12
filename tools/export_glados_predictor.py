"""Export GLaDOS weights into a Burn-oriented interchange checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PREDICTORS = ("dur_pred", "pitch_cond_pred", "pitch_pred", "energy_pred")


def split_gru_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    hidden = tensor.shape[0] // 3
    if tensor.shape[0] != hidden * 3:
        raise ValueError(f"expected a 3-gate GRU tensor, got shape {tuple(tensor.shape)}")
    reset, update, new = tensor.split(hidden, dim=0)
    return update, reset, new


def burn_gate_name(module: str, direction: str, gate: str, transform: str, suffix: str) -> str:
    return f"{module}.{direction}.{gate}_gate.{transform}.{suffix}"


def split_lstm_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    hidden = tensor.shape[0] // 4
    if tensor.shape[0] != hidden * 4:
        raise ValueError(f"expected a 4-gate LSTM tensor, got shape {tuple(tensor.shape)}")
    input_gate, forget_gate, cell_gate, output_gate = tensor.split(hidden, dim=0)
    return input_gate, forget_gate, output_gate, cell_gate


def burn_lstm_gate_name(direction: str, gate: str, transform: str, suffix: str) -> str:
    return f"lstm.{direction}.{gate}_gate.{transform}.{suffix}"


def remap_non_recurrent_key(key: str) -> str:
    key = key.replace(".bnorm.weight", ".bnorm.gamma")
    key = key.replace(".bnorm.bias", ".bnorm.beta")
    key = key.replace(".W1.", ".w1.")
    key = key.replace(".W2.", ".w2.")
    return key


def export_recurrent_tensor(
    exported: dict[str, torch.Tensor], key: str, value: torch.Tensor, prefix: str, module: str = "rnn"
) -> None:
    local_key = key.removeprefix(prefix)
    rnn_key = local_key.removeprefix(module + ".")
    direction = "reverse" if rnn_key.endswith("_reverse") else "forward"
    direction_suffix = "_reverse" if direction == "reverse" else ""
    for torch_name in (
        "weight_ih_l0",
        "weight_hh_l0",
        "bias_ih_l0",
        "bias_hh_l0",
    ):
        if rnn_key == torch_name + direction_suffix:
            transform = (
                "input_transform"
                if torch_name.startswith("weight_ih") or torch_name.startswith("bias_ih")
                else "hidden_transform"
            )
            suffix = "weight" if torch_name.startswith("weight") else "bias"
            for gate, gate_tensor in zip(
                ("update", "reset", "new"), split_gru_tensor(value), strict=True
            ):
                exported[burn_gate_name(module, direction, gate, transform, suffix)] = gate_tensor
            return
    raise ValueError(f"unexpected GRU key {key}")


def export_predictor_state(
    exported: dict[str, torch.Tensor], key: str, value: torch.Tensor, prefix: str
) -> None:
    local_key = key.removeprefix(prefix)
    if local_key.startswith("rnn."):
        export_recurrent_tensor(exported, key, value, prefix)
        return
    exported[remap_non_recurrent_key(local_key)] = value


def export_predictor(source: Path, output: Path, predictor: str) -> None:
    if predictor not in PREDICTORS:
        raise ValueError(f"unsupported predictor {predictor!r}; choose from {PREDICTORS}")

    module = torch.jit.load(str(source), map_location="cpu")
    state = module.state_dict()
    prefix = predictor + "."
    exported: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        if key.endswith("num_batches_tracked"):
            continue
        export_predictor_state(exported, key, value, prefix)

    expected = {
        key
        for key in state
        if key.startswith(prefix) and not key.endswith("num_batches_tracked")
    }
    if not exported:
        raise ValueError(f"no tensors found for predictor {predictor!r} in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    print(f"exported {len(exported)} tensors for {predictor} to {output}")
    print(f"consumed {len(expected)} upstream tensors (excluding BatchNorm counters)")


def export_full(source: Path, output: Path) -> None:
    module = torch.jit.load(str(source), map_location="cpu")
    state = module.state_dict()
    exported: dict[str, torch.Tensor] = {}
    skipped = {"step"}
    for key, value in state.items():
        if key in skipped or key.endswith("num_batches_tracked"):
            continue
        predictor = next((name for name in PREDICTORS if key.startswith(name + ".")), None)
        if predictor is not None:
            predictor_exported: dict[str, torch.Tensor] = {}
            export_predictor_state(predictor_exported, key, value, predictor + ".")
            for local_key, tensor in predictor_exported.items():
                exported[f"{predictor}.{local_key}"] = tensor
            continue
        cbhg = next((name for name in ("prenet", "postnet") if key.startswith(name + ".rnn.")), None)
        if cbhg is not None:
            cbhg_exported: dict[str, torch.Tensor] = {}
            export_recurrent_tensor(cbhg_exported, key, value, cbhg + ".")
            for local_key, tensor in cbhg_exported.items():
                exported[f"{cbhg}.{local_key}"] = tensor
            continue
        if key.startswith("lstm."):
            local_key = key.removeprefix("lstm.")
            direction = "reverse" if local_key.endswith("_reverse") else "forward"
            direction_suffix = "_reverse" if direction == "reverse" else ""
            for torch_name in (
                "weight_ih_l0",
                "weight_hh_l0",
                "bias_ih_l0",
                "bias_hh_l0",
            ):
                if local_key == torch_name + direction_suffix:
                    transform = (
                        "input_transform"
                        if torch_name.startswith("weight_ih") or torch_name.startswith("bias_ih")
                        else "hidden_transform"
                    )
                    suffix = "weight" if torch_name.startswith("weight") else "bias"
                    for gate, gate_tensor in zip(
                        ("input", "forget", "output", "cell"),
                        split_lstm_tensor(value),
                        strict=True,
                    ):
                        exported[burn_lstm_gate_name(direction, gate, transform, suffix)] = gate_tensor
                    break
            else:
                raise ValueError(f"unexpected LSTM key {key}")
            continue
        exported[remap_non_recurrent_key(key)] = value
    if not exported:
        raise ValueError(f"no tensors found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    print(f"exported {len(exported)} tensors for full GLaDOS acoustic model to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--predictor", choices=PREDICTORS)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full == (args.predictor is not None):
        parser.error("choose exactly one of --predictor or --full")
    if args.full:
        export_full(args.source, args.output)
    else:
        export_predictor(args.source, args.output, args.predictor)


if __name__ == "__main__":
    main()
