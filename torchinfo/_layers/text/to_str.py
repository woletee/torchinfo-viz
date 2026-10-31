from __future__ import annotations


def macs_to_str(self, reached_max_depth: bool) -> str:
    """Convert MACs to string."""
    if self.macs <= 0:
        return "--"
    if self.is_leaf_layer:
        return f"{self.macs:,}"
    if reached_max_depth:
        sum_child_macs = sum(
            (child.macs for child in self.children if child.is_leaf_layer)
        )
        return f"{sum_child_macs:,}"
    return "--"


def num_params_to_str(self, reached_max_depth: bool) -> str:
    """Convert num_params to string."""
    if self.num_params == 0:
        return "--"
    if self.is_recursive:
        return "(recursive)"
    if reached_max_depth or self.is_leaf_layer:
        param_count_str = f"{self.num_params:,}"
        return param_count_str if self.trainable_params else f"({param_count_str})"
    leftover_params = self.leftover_params()
    return f"{leftover_params:,}" if leftover_params > 0 else "--"
