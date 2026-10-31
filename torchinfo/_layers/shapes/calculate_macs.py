from __future__ import annotations
from torchinfo._layers.utils import prod


def calculate_macs(self) -> None:
    """Set MACs using the module's parameters and layer's output size, which is"""
    for name, param in self.module.named_parameters():
        cur_params, name = self.get_param_count(self.module, name, param)
        if name in ("weight", "bias"):
            if "Conv" in self.class_name:
                self.macs += int(
                    cur_params * prod(self.output_size[:1] + self.output_size[2:])
                )
            elif "Linear" in self.class_name:
                self.macs += int(cur_params * prod(self.output_size[:-1]))
            else:
                self.macs += self.output_size[0] * cur_params
        elif "weight" in name or "bias" in name:
            self.macs += prod(self.output_size[:2]) * cur_params
