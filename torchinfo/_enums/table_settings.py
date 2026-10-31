from __future__ import annotations
from enum import StrEnum, unique


@unique
class ColumnSettings(StrEnum):
    """Enum containing all available column settings."""

    __slots__ = ()
    KERNEL_SIZE = "kernel_size"
    GROUPS = "groups"
    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    NUM_PARAMS = "num_params"
    PARAMS_PERCENT = "params_percent"
    MULT_ADDS = "mult_adds"
    TRAINABLE = "trainable"


@unique
class RowSettings(StrEnum):
    """Enum containing all available row settings."""

    __slots__ = ()
    DEPTH = "depth"
    VAR_NAMES = "var_names"
    ASCII_ONLY = "ascii_only"
    HIDE_RECURSIVE_LAYERS = "hide_recursive_layers"
