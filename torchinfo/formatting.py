from __future__ import annotations
from ._constants.display import HEADER_TITLES
from ._constants.display import CONVERSION_FACTORS


class FormattingOptions:
    """Class that holds information about formatting the table output."""

    from torchinfo._formatting.initialize import __init__
    from torchinfo._formatting.strings import str_

    str_ = staticmethod(str_)
    from torchinfo._formatting.strings import get_start_str
    from torchinfo._formatting.widths import set_layer_name_width
    from torchinfo._formatting.widths import get_total_width
    from torchinfo._formatting.rows import format_row
    from torchinfo._formatting.rows import header_row
    from torchinfo._formatting.layer_info_to_row import layer_info_to_row
    from torchinfo._formatting.layers_to_str import layers_to_str
