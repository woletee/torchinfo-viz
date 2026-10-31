from __future__ import annotations
from torchinfo._constants.defaults import DEFAULT_COLUMN_NAMES
from torchinfo._constants.defaults import DEFAULT_ROW_SETTINGS
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._enums.run_settings import Mode
from torchinfo._enums.table_settings import RowSettings
import sys


def summary_options(args):
    has_input = args.input_data is not None or args.input_size is not None
    if args.col_names is None:
        columns = DEFAULT_COLUMN_NAMES if has_input else (ColumnSettings.NUM_PARAMS,)
    else:
        columns = tuple((ColumnSettings(name) for name in args.col_names))
    rows = (
        DEFAULT_ROW_SETTINGS
        if args.row_settings is None
        else {RowSettings(name) for name in args.row_settings}
    )
    model_mode = Mode(args.mode)
    verbose = args.verbose
    if verbose is None:
        verbose = 0 if hasattr(sys, "ps1") and sys.ps1 else 1
    return (columns, rows, model_mode, verbose)
