from __future__ import annotations
from enum import StrEnum
from enum import unique


@unique
class Units(StrEnum):
    """Enum containing all available bytes units."""

    __slots__ = ()
    AUTO = "auto"
    KILOBYTES = "k"
    MEGABYTES = "M"
    GIGABYTES = "G"
    TERABYTES = "T"
    NONE = ""
