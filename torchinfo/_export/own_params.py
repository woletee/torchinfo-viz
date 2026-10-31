from __future__ import annotations
from typing import Any


def own_params(layer: Any, seen: set[int], called: set[int]) -> int:
    """Parameters this layer owns directly, counting each tensor only once.

    Modules registered under several parents (e.g. MONAI's DynUNet) would
    otherwise be counted again under every parent. Submodules that never run
    as modules (e.g. MultiheadAttention.out_proj, used via F.linear) are
    folded into this layer, since no row of their own exists.
    """
    count = 0
    for param_id, num_params, _, _ in layer.get_direct_param_ids():
        if param_id not in seen:
            seen.add(param_id)
            count += num_params
    for module in uncalled_modules(layer.module, called):
        for param in module.parameters(recurse=False):
            if id(param) not in seen:
                seen.add(id(param))
                count += param.numel()
    return count


def uncalled_modules(module: Any, called: set[int]) -> list[Any]:
    """Descendants of `module` reachable without passing a module that ran."""
    found = [child for child in module.children() if id(child) not in called]
    return found + [m for child in found for m in uncalled_modules(child, called)]
