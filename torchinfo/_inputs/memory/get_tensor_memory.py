from __future__ import annotations
from torchinfo._inputs.sparse_members import sparse_members
import sys
import torch


def get_tensor_memory(data: torch.Tensor) -> int:
    """Calculate storage memory, including sparse tensor members."""
    try:
        storage = (
            data.untyped_storage()
            if hasattr(data, "untyped_storage")
            else data.storage()
        )
        return sys.getsizeof(storage)
    except NotImplementedError:
        return sum((get_tensor_memory(member) for member in sparse_members(data)))
