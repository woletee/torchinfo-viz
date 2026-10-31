from __future__ import annotations


def leftover_params(self) -> int:
    """Leftover params are the number of params this current layer has that are not"""
    return self._leftover("num_params")


def leftover_trainable_params(self) -> int:
    return self._leftover("trainable_params")


def leftover_param_bytes(self) -> int:
    """Leftover param bytes are the bytes of this layer's params that are not"""
    return self._leftover("param_bytes")
