from __future__ import annotations


_cached_forward_pass = {}


def clear_cached_forward_pass() -> None:
    """Clear the forward pass cache."""
    _cached_forward_pass.clear()
