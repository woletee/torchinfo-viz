from __future__ import annotations
from torchinfo._statistics.accumulate_totals import accumulate_totals
from torchinfo._statistics.mark_shared_parameters import mark_shared_parameters
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.formatting import FormattingOptions
    from torchinfo.layer_info import LayerInfo


def __init__(
    self,
    summary_list: list[LayerInfo],
    input_size: Any,
    total_input_size: int,
    formatting: FormattingOptions,
) -> None:
    self.summary_list = summary_list
    self.input_size = input_size
    self.formatting = formatting
    self.total_input = total_input_size
    self.total_mult_adds = 0
    self.total_params = self.trainable_params = 0
    self.total_param_bytes = self.total_output_bytes = 0
    accumulate_totals(self, summary_list)
    mark_shared_parameters(summary_list)
    self.formatting.set_layer_name_width(summary_list)
