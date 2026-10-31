from __future__ import annotations
from torchinfo._constants.input_types import INPUT_DATA_TYPE
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._execution.setup.validate_input_args import validate_input_args
from torchinfo._execution.setup.warn_precision import warn_precision
import torch


def validate_user_params(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    col_names: tuple[ColumnSettings, ...],
    col_width: int,
    device: torch.device | None,
    dtypes: list[torch.dtype] | None,
    verbose: int,
) -> None:
    """Raise exceptions if the user's input is invalid."""
    if col_width <= 0:
        raise ValueError(f"Column width must be greater than 0: col_width={col_width}")
    if verbose not in (0, 1, 2):
        raise ValueError(
            "Verbose must be either 0 (quiet), 1 (default), or 2 (verbose)."
        )
    validate_input_args(input_data, input_size, col_names)
    warn_precision(input_size, device, dtypes)
