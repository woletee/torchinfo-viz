from __future__ import annotations


try:
    from torch.nn.parameter import is_lazy
except ImportError:

    def is_lazy(param):
        return False


def trainable(self) -> str:
    """Checks if the module is trainable. Returns:"""
    if self.num_params == 0:
        return "--"
    if self.trainable_params == 0:
        return "False"
    if self.num_params == self.trainable_params:
        return "True"
    if self.num_params > self.trainable_params:
        return "Partial"
    raise RuntimeError("Unreachable trainable calculation.")
