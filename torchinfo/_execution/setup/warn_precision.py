from __future__ import annotations
import torch
import warnings


def warn_precision(input_size, device, dtypes):
    if dtypes is not None and any(
        (dtype in (torch.float16, torch.bfloat16) for dtype in dtypes)
    ):
        if input_size is not None:
            warnings.warn(
                "Half precision is not officially supported with the `input_size` parameter, as it may output incorrect size estimation results. We recommend passing `input_data` directly.",
                stacklevel=2,
            )
        if device is not None and device.type == "cpu":
            warnings.warn(
                "Half precision is not supported on cpu. Set the `device` field or pass `input_data` using the correct device.",
                stacklevel=2,
            )
