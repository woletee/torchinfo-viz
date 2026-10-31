from __future__ import annotations
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from functools import partial
from typing import Any
import numpy as np
import torch


def traverse_input_data(
    data: Any, action_fn: Callable[..., Any], aggregate_fn: Callable[..., Any]
) -> Any:
    """Apply a tensor action recursively, retaining collection structure."""
    recurse = partial(
        traverse_input_data, action_fn=action_fn, aggregate_fn=aggregate_fn
    )
    if isinstance(data, torch.Tensor):
        return action_fn(data)
    if isinstance(data, np.ndarray):
        result = action_fn(torch.from_numpy(data))
        return data if isinstance(result, torch.Tensor) else result
    if isinstance(data, Mapping):
        return aggregate_fn(data)({k: recurse(v) for k, v in data.items()})
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return aggregate_fn(data)(*(recurse(v) for v in data))
    if isinstance(data, Iterable) and (not isinstance(data, str)):
        return aggregate_fn(data)([recurse(v) for v in data])
    return data
