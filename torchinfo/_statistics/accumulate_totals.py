from __future__ import annotations


def accumulate_totals(statistics, layers):
    """Root parameter totals deduplicate tied weights and include unused modules."""
    for layer in layers:
        if layer.is_leaf_layer:
            statistics.total_mult_adds += layer.macs
            if layer.num_params > 0:
                statistics.total_output_bytes += layer.output_bytes * 2
        if layer.parent_info is None:
            statistics.total_params += layer.num_params
            statistics.trainable_params += layer.trainable_params
            statistics.total_param_bytes += layer.param_bytes
