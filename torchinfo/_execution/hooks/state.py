from __future__ import annotations


class HookState:
    """Mutable state shared by the hooks of one model traversal."""

    def __init__(self, batch_dim):
        self.batch_dim = batch_dim
        self.summary_list = []
        self.layer_ids = set()
        self.global_layer_info = {}
        self.hooks = {}
        self.contexts = {}
        self.module_stack = []
