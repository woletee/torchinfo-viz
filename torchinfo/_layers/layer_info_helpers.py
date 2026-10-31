from __future__ import annotations


class LayerInfoHelpers:
    """Static size helpers and string formatting methods used by LayerInfo."""

    from torchinfo._layers.shapes.calculate_size import calculate_size

    calculate_size = staticmethod(calculate_size)
    from torchinfo._layers.params.get_param_count import get_param_count

    get_param_count = staticmethod(get_param_count)
    from torchinfo._layers.shapes.conv_info import get_kernel_size

    get_kernel_size = staticmethod(get_kernel_size)
    from torchinfo._layers.shapes.conv_info import get_groups

    get_groups = staticmethod(get_groups)
    from torchinfo._layers.text.to_str import macs_to_str
    from torchinfo._layers.text.to_str import num_params_to_str
    from torchinfo._layers.params.params_percent import params_percent
    from torchinfo._layers.params._leftover import _leftover
    from torchinfo._layers.params.leftover import leftover_params
    from torchinfo._layers.params.leftover import leftover_trainable_params
    from torchinfo._layers.params.leftover import leftover_param_bytes
