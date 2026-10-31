from __future__ import annotations
from torchinfo._statistics.format_output_num import format_output_num


def parameter_totals(statistics):
    units = statistics.formatting.params_count_units
    total = format_output_num(statistics.total_params, units, False)
    trainable = format_output_num(statistics.trainable_params, units, False)
    frozen = format_output_num(
        statistics.total_params - statistics.trainable_params, units, False
    )
    return f"Total params{total}\nTrainable params{trainable}\nNon-trainable params{frozen}\n"
