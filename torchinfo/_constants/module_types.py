from __future__ import annotations
from torch.jit import ScriptModule
import torch


WRAPPER_MODULES = (ScriptModule,)


LAYER_MODULES = (torch.nn.MultiheadAttention,)


LayerContext = tuple[str, int, "LayerInfo | None"]
