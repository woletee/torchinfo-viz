from __future__ import annotations
from torchinfo._constants.display import CONVERSION_FACTORS
from torchinfo._enums.Units import Units


def float_to_megabytes(num: int) -> float:
    """Converts a number (assume floats, 4 bytes each) to megabytes."""
    return num * 4 / 1000000.0


def to_megabytes(num: int) -> float:
    """Converts bytes to megabytes."""
    return num / 1000000.0


def to_readable(num: float, units: Units = Units.AUTO) -> tuple[Units, float]:
    """Converts a number to millions, billions, or trillions."""
    if units == Units.AUTO:
        if num >= 1000000000000.0:
            return (Units.TERABYTES, num / 1000000000000.0)
        if num >= 1000000000.0:
            return (Units.GIGABYTES, num / 1000000000.0)
        if num >= 1000000.0:
            return (Units.MEGABYTES, num / 1000000.0)
        if num >= 1000.0:
            return (Units.KILOBYTES, num / 1000.0)
        return (Units.NONE, num)
    return (units, num / CONVERSION_FACTORS[units])
