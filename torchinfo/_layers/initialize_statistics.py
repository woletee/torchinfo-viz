from __future__ import annotations


def initialize_statistics(self):
    self.executed = False
    self.contains_lazy_param = False
    self.is_recursive = False
    self.input_size: list[int] = []
    self.output_size: list[int] = []
    self.kernel_size = self.get_kernel_size(self.module)
    self.groups = self.get_groups(self.module)
    self.trainable_params = 0
    self.num_params = 0
    self.param_bytes = 0
    self.output_bytes = 0
    self.macs = 0
