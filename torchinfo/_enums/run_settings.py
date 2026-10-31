from __future__ import annotations
from enum import StrEnum, unique, IntEnum


@unique
class Mode(StrEnum):
    """Enum containing all model modes."""

    __slots__ = ()
    TRAIN = "train"
    EVAL = "eval"
    SAME = "same"


@unique
class Verbosity(IntEnum):
    """Contains verbosity levels."""

    QUIET, DEFAULT, VERBOSE = (0, 1, 2)
