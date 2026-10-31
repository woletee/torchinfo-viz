from __future__ import annotations
from collections.abc import Iterable
from torch import nn
from torchinfo._constants.input_types import INPUT_DATA_TYPE
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from torchinfo._execution.build_summary import build_summary
from torchinfo._execution.summary_args import SummaryArgs
from torchinfo.model_statistics import ModelStatistics
from typing import Any
import torch


def summary(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE | None = None,
    input_data: INPUT_DATA_TYPE | None = None,
    batch_dim: int | None = None,
    cache_forward_pass: bool | None = None,
    col_names: Iterable[str] | None = None,
    col_width: int = 25,
    depth: int = 3,
    device: torch.device | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str = "same",
    row_settings: Iterable[str] | None = None,
    verbose: int | None = None,
    **kwargs: Any,
) -> ModelStatistics:
    """Summarize a model. See README.md for arguments and usage examples."""
    return build_summary(SummaryArgs(**locals()))
