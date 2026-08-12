"""Extract the frozen upstream GPU vocoder constants into a state dictionary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def tensor_constant(graph_value: torch.Value) -> torch.Tensor:
    value = graph_value.toIValue()
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"expected tensor constant for {graph_value.debugName()}")
    return value.detach().cpu()


def add_pair(
    exported: dict[str, torch.Tensor],
    graph_nodes: list[torch.Node],
    index: int,
    weight_name: str,
    bias_name: str,
) -> int:
    node = graph_nodes[index]
    if node.kind() not in ("aten::conv1d", "aten::conv_transpose1d"):
        raise ValueError(f"expected convolution node at index {index}, got {node.kind()}")
    inputs = list(node.inputs())
    exported[weight_name] = tensor_constant(inputs[1])
    exported[bias_name] = tensor_constant(inputs[2])
    return index + 1


def export(source: Path, output: Path) -> None:
    module = torch.jit.load(str(source), map_location="cpu")
    convolution_nodes = [
        node
        for node in module.graph.nodes()
        if node.kind() in ("aten::conv1d", "aten::conv_transpose1d")
    ]
    if len(convolution_nodes) != 78:
        raise ValueError(f"expected 78 convolution nodes, found {len(convolution_nodes)}")

    exported: dict[str, torch.Tensor] = {}
    index = 0
    index = add_pair(exported, convolution_nodes, index, "conv_pre.weight", "conv_pre.bias")
    for stage, (input_channels, output_channels) in enumerate(
        ((512, 256), (256, 128), (128, 64), (64, 32))
    ):
        index = add_pair(
            exported,
            convolution_nodes,
            index,
            f"ups.{stage}.weight",
            f"ups.{stage}.bias",
        )
        for block in range(3):
            for layer in range(3):
                index = add_pair(
                    exported,
                    convolution_nodes,
                    index,
                    f"resblocks.{stage}.{block}.convs1.{layer}.weight",
                    f"resblocks.{stage}.{block}.convs1.{layer}.bias",
                )
                index = add_pair(
                    exported,
                    convolution_nodes,
                    index,
                    f"resblocks.{stage}.{block}.convs2.{layer}.weight",
                    f"resblocks.{stage}.{block}.convs2.{layer}.bias",
                )
    index = add_pair(exported, convolution_nodes, index, "conv_post.weight", "conv_post.bias")
    if index != len(convolution_nodes):
        raise AssertionError(f"consumed {index} convolution nodes")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    print(f"exported {len(exported)} vocoder tensors to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == "__main__":
    main()
