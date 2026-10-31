from __future__ import annotations
from typing import Any

# Ops that only reshape, move or view a tensor: they keep the producer's node.
PASSTHROUGH = frozenset(
    "view reshape flatten permute transpose t contiguous squeeze unsqueeze getitem "
    "get expand expand_as to float double half bfloat16 type clone detach cpu cuda "
    "chunk split unbind narrow select view_as".split()
)
# Operator dunders that mean the same op, e.g. x + y calls __add__ or __radd__.
ALIASES = dict(
    pair.split(":")
    for pair in "radd:add iadd:add rmul:mul imul:mul rsub:sub isub:sub "
    "truediv:div rtruediv:div".split()
)


def op_name(func: Any) -> str:
    """Readable name of a torch function or Tensor method: __add__ -> add."""
    name = str(getattr(func, "__name__", func)).strip("_")
    return ALIASES.get(name, name)
