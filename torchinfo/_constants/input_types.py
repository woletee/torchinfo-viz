from __future__ import annotations
from collections.abc import Sequence, Mapping, Iterable
from typing import Any
import torch
import numpy as np


INPUT_SIZE_TYPE = Sequence[int | Sequence[Any] | torch.Size]


INPUT_DATA_TYPE = torch.Tensor | np.ndarray | Sequence[Any] | Mapping[str, Any]


CORRECTED_INPUT_SIZE_TYPE = list[Sequence[Any] | torch.Size]


CORRECTED_INPUT_DATA_TYPE = Iterable[Any] | Mapping[Any, Any] | None


DETECTED_INPUT_OUTPUT_TYPES = Sequence[Any] | Mapping[Any, torch.Tensor] | torch.Tensor
