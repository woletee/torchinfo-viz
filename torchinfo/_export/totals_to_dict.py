from __future__ import annotations
from typing import Any


def totals_to_dict(statistics: Any) -> dict[str, int]:
    """Model-wide totals of a ModelStatistics, as plain numbers."""
    return {
        "total_params": statistics.total_params,
        "trainable_params": statistics.trainable_params,
        "total_mult_adds": statistics.total_mult_adds,
        "total_param_bytes": statistics.total_param_bytes,
        "total_output_bytes": statistics.total_output_bytes,
        "total_input_bytes": statistics.total_input,
    }
