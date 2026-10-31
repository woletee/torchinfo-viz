from __future__ import annotations
from torch.overrides import TorchFunctionMode
from torchinfo._graph.state.graph_state import GraphState
from torchinfo._graph.recording.record_op import record_op
from typing import Any


class RecordingMode(TorchFunctionMode):
    """Sees every torch function and Tensor method called during the forward."""

    def __init__(self, state: GraphState) -> None:
        super().__init__()
        self.state = state

    def __torch_function__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        if not self.state.failed:
            try:
                record_op(self.state, func, args, kwargs, output)
            except Exception:
                self.state.failed = True
        return output
