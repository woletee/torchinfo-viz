from __future__ import annotations
from torchinfo._constants.input_types import INPUT_DATA_TYPE
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from torchinfo._constants.defaults import REQUIRES_INPUT
from torchinfo._enums.table_settings import ColumnSettings


def validate_input_args(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    col_names: tuple[ColumnSettings, ...],
) -> None:
    """Raise exceptions if the input arguments conflict with the columns."""
    both_input_specified = input_data is not None and input_size is not None
    if both_input_specified:
        raise RuntimeError("Only one of (input_data, input_size) should be specified.")
    neither_input_specified = input_data is None and input_size is None
    not_allowed = set(col_names) & REQUIRES_INPUT
    if neither_input_specified and not_allowed:
        raise ValueError(
            f"You must pass input_data or input_size in order to use columns: {not_allowed}"
        )
