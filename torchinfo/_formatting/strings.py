from __future__ import annotations
from typing import Any


def str_(val: Any) -> str:
    return str(val) if val else "--"


def get_start_str(self, depth: int) -> str:
    """This function should handle all ascii/non-ascii-related characters."""
    if depth == 0:
        return ""
    if depth == 1:
        return "+ " if self.ascii_only else "├─"
    return ("|    " if self.ascii_only else "│    ") * (depth - 1) + (
        "+ " if self.ascii_only else "└─"
    )
