from __future__ import annotations


def mark_shared_parameters(layers):
    """Mark rows whose directly owned parameters were all counted earlier."""
    seen = set()
    for layer in layers:
        parameters = layer.get_direct_param_ids()
        contributed = False
        for param_id, _, _, _ in parameters:
            if param_id not in seen:
                seen.add(param_id)
                contributed = True
        if not contributed and any((count for _, count, _, _ in parameters)):
            layer.is_recursive = True
