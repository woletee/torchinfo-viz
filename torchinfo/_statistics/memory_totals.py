from __future__ import annotations
from torchinfo._statistics.format_output_num import format_output_num


def memory_totals(statistics, divider):
    units = statistics.formatting.params_size_units
    macs = format_output_num(
        statistics.total_mult_adds, statistics.formatting.macs_units, False
    )
    inputs = format_output_num(statistics.total_input, units, True)
    outputs = format_output_num(statistics.total_output_bytes, units, True)
    parameters = format_output_num(statistics.total_param_bytes, units, True)
    total = format_output_num(
        statistics.total_input
        + statistics.total_output_bytes
        + statistics.total_param_bytes,
        units,
        True,
    )
    return f"Total mult-adds{macs}\n{divider}\nInput size{inputs}\nForward/backward pass size{outputs}\nParams size{parameters}\nEstimated Total Size{total}\n"
