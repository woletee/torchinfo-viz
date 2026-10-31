from __future__ import annotations
from ._layers.shapes.nested_list_size import nested_list_size
from ._layers.utils import prod
from ._layers.utils import rgetattr
from ._layers.tree.get_children_layers import get_children_layers
from ._layers.layer_info_helpers import LayerInfoHelpers


class LayerInfo(LayerInfoHelpers):
    """Class that holds information about a layer module."""

    from torchinfo._layers.initialize import __init__
    from torchinfo._layers.text.representation import __repr__
    from torchinfo._layers.params.params import trainable

    trainable = property(trainable)
    from torchinfo._layers.text.names import get_layer_name
    from torchinfo._layers.params.calculate_num_params import calculate_num_params
    from torchinfo._layers.params.get_direct_param_ids import get_direct_param_ids
    from torchinfo._layers.shapes.calculate_macs import calculate_macs
    from torchinfo._layers.tree.recursion import check_recursive
