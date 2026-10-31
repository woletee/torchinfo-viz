from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from torch import nn
from torchinfo._constants.input_types import INPUT_DATA_TYPE
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from typing import Any
import torch


@dataclass
class SummaryArgs:
    """The arguments a user passed to summary()."""

    model: nn.Module
    input_size: INPUT_SIZE_TYPE | None
    input_data: INPUT_DATA_TYPE | None
    batch_dim: int | None
    cache_forward_pass: bool | None
    col_names: Iterable[str] | None
    col_width: int
    depth: int
    device: torch.device | None
    dtypes: list[torch.dtype] | None
    mode: str
    row_settings: Iterable[str] | None
    verbose: int | None
    kwargs: dict[str, Any]
