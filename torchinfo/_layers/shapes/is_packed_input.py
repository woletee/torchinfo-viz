from __future__ import annotations


def is_packed_input(inputs):
    return (
        isinstance(inputs, (list, tuple))
        and bool(inputs)
        and hasattr(inputs[0], "data")
        and hasattr(inputs[0].data, "size")
    )
