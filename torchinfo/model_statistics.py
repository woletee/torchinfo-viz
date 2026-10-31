from __future__ import annotations


class ModelStatistics:
    """Class for storing results of the summary."""

    from torchinfo._statistics.initialize import __init__
    from torchinfo._statistics.representation import __repr__
    from torchinfo._export.to_dict import to_dict
    from torchinfo._export.to_json import to_json
    from torchinfo._export.to_html import to_html
    from torchinfo._export.show import show
    from torchinfo._statistics.units import float_to_megabytes

    float_to_megabytes = staticmethod(float_to_megabytes)
    from torchinfo._statistics.units import to_megabytes

    to_megabytes = staticmethod(to_megabytes)
    from torchinfo._statistics.units import to_readable

    to_readable = staticmethod(to_readable)
    from torchinfo._statistics.format_output_num import format_output_num

    format_output_num = staticmethod(format_output_num)
