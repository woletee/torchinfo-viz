from __future__ import annotations
from torch import nn
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._layers.text.names import get_class_name
from torchinfo._layers.initialize_statistics import initialize_statistics
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def __init__(
    self,
    var_name: str,
    module: nn.Module,
    depth: int,
    parent_info: LayerInfo | None = None,
) -> None:
    self.layer_id = id(module)
    self.module = module
    self.class_name = get_class_name(module)
    self.inner_layers: dict[str, dict[ColumnSettings, Any]] = {}
    self.depth = depth
    self.depth_index: int | None = None
    self.children: list[LayerInfo] = []
    self.parent_info = parent_info
    self.var_name = var_name
    self.is_leaf_layer = not any(self.module.children())
    initialize_statistics(self)
