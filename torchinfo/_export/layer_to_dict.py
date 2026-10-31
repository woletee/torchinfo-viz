from __future__ import annotations
from torchinfo._export.own_params import own_params
from typing import Any


def layer_to_dict(
    layer: Any, index: int, parent: int | None, seen: set[int], called: set[int]
) -> dict[str, Any]:
    """Serialize one LayerInfo row; `seen` and `called` hold ids already done."""
    return {
        "id": index,
        "parent": parent,
        "name": layer.class_name,
        "var_name": layer.var_name,
        "depth": layer.depth,
        "depth_index": layer.depth_index,
        "input_size": layer.input_size,
        "output_size": layer.output_size,
        "kernel_size": layer.kernel_size,
        "num_params": layer.num_params,
        "self_params": own_params(layer, seen, called),
        "trainable": layer.trainable,
        "macs": layer.macs,
        "param_bytes": layer.param_bytes,
        "output_bytes": layer.output_bytes,
        "flags": {"leaf": layer.is_leaf_layer, "recursive": layer.is_recursive},
    }
