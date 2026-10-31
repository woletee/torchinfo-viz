from __future__ import annotations
from collections.abc import Mapping
from torchinfo._enums.run_settings import Mode


def call_model(model, inputs, kwargs):
    if isinstance(inputs, (list, tuple)):
        model(*inputs, **kwargs)
    elif isinstance(inputs, Mapping):
        model(**inputs, **kwargs)
    else:
        raise ValueError("Unknown input type")


def set_model_mode(model, mode):
    if mode == Mode.TRAIN:
        model.train()
    elif mode == Mode.EVAL:
        model.eval()
    elif mode != Mode.SAME:
        raise RuntimeError(
            f"Specified model mode ({list(Mode)}) not recognized: {mode}"
        )
