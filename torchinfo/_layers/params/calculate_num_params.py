from __future__ import annotations
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._layers.params.inner_parameter_row import inner_parameter_row
from torchinfo._layers.params.params import is_lazy


def calculate_num_params(self) -> None:
    """Collect parameters, pruning masks, trainability, and inner-layer rows."""
    self.num_params = self.param_bytes = self.trainable_params = 0
    self.inner_layers = {}
    final_name = ""
    for name, param in self.module.named_parameters():
        if is_lazy(param):
            self.contains_lazy_param = True
            continue
        count, name = self.get_param_count(self.module, name, param)
        self.param_bytes += param.element_size() * count
        self.num_params += count
        if param.requires_grad:
            self.trainable_params += count
        if self.parent_info is not None or "." not in name:
            self.inner_layers[name] = inner_parameter_row(name, param, count)
            final_name = name
    if self.inner_layers:
        self.inner_layers[final_name][ColumnSettings.NUM_PARAMS] = (
            f"└─{self.inner_layers[final_name][ColumnSettings.NUM_PARAMS][2:]}"
        )
