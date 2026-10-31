from __future__ import annotations
from torchinfo._layers.params.params import is_lazy


def get_direct_param_ids(self) -> list[tuple[int, int, int, bool]]:
    """Return (param_tensor_id, num_params, param_bytes, is_trainable) for each"""
    contributions = []
    for name, param in self.module.named_parameters(recurse=False):
        if is_lazy(param):
            continue
        cur_params, _ = self.get_param_count(self.module, name, param)
        contributions.append(
            (
                id(param),
                cur_params,
                param.element_size() * cur_params,
                param.requires_grad,
            )
        )
    return contributions
