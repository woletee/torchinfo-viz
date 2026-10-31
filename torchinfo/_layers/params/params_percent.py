from __future__ import annotations


def params_percent(
    self, total_params: int, reached_max_depth: bool, precision: int = 2
) -> str:
    """Convert num_params to string."""
    spacing = 5
    zero = f"{' ' * spacing}--"
    if total_params == 0:
        return zero
    if self.is_recursive:
        return "(recursive)"
    params = (
        self.num_params
        if reached_max_depth or self.is_leaf_layer
        else self.leftover_params()
    )
    if params == 0:
        return zero
    return f"{params / total_params:>{precision + spacing}.{precision}%}"
