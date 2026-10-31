from __future__ import annotations
from torchinfo._enums.Units import Units
from torchinfo._statistics.units import to_readable


def format_output_num(num: int, units: Units, is_bytes: bool) -> str:
    units_used, converted_num = to_readable(num, units)
    if isinstance(converted_num, float) and converted_num.is_integer():
        converted_num = int(converted_num)
    units_display = (
        ""
        if units_used == Units.NONE
        else f" ({units_used.value}{('B' if is_bytes else '')})"
    )
    fmt = "d" if isinstance(converted_num, int) else ".2f"
    return f"{units_display}: {converted_num:,{fmt}}"
