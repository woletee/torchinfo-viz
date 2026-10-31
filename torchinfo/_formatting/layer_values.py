from __future__ import annotations
from torchinfo._enums.table_settings import ColumnSettings


def layer_values(self, layer_info, reached_max_depth, total_params):
    return {
        ColumnSettings.KERNEL_SIZE: self.str_(layer_info.kernel_size),
        ColumnSettings.GROUPS: self.str_(layer_info.groups),
        ColumnSettings.INPUT_SIZE: self.str_(layer_info.input_size),
        ColumnSettings.OUTPUT_SIZE: self.str_(layer_info.output_size),
        ColumnSettings.NUM_PARAMS: layer_info.num_params_to_str(reached_max_depth),
        ColumnSettings.PARAMS_PERCENT: layer_info.params_percent(
            total_params, reached_max_depth
        ),
        ColumnSettings.MULT_ADDS: layer_info.macs_to_str(reached_max_depth),
        ColumnSettings.TRAINABLE: self.str_(layer_info.trainable),
    }
