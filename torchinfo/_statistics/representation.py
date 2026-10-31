from __future__ import annotations
from torchinfo._statistics.memory_totals import memory_totals
from torchinfo._statistics.parameter_totals import parameter_totals


def __repr__(self) -> str:
    """Render the summary table and aggregate statistics."""
    divider = "=" * self.formatting.get_total_width()
    layers = self.formatting.layers_to_str(self.summary_list, self.total_params)
    result = f"{divider}\n{self.formatting.header_row()}{divider}\n{layers}{divider}\n"
    result += parameter_totals(self)
    if self.input_size:
        result += memory_totals(self, divider)
    return result + divider
