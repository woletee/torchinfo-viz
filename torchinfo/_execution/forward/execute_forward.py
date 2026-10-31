from __future__ import annotations
from torchinfo._execution.forward.run_model import call_model
from torchinfo._execution.forward.run_model import set_model_mode
import torch


def execute_forward(model, x, kwargs, device, mode, layers, hooks):
    saved_mode = model.training
    try:
        set_model_mode(model, mode)
        with torch.no_grad():
            call_model(model if device is None else model.to(device), x, kwargs)
    except Exception as error:
        executed = [layer for layer in layers if layer.executed]
        raise RuntimeError(
            f"Failed to run torchinfo. See above stack traces for more details. Executed layers up to: {executed}"
        ) from error
    finally:
        for pre_hook, hook in hooks.values():
            pre_hook.remove()
            hook.remove()
        model.train(saved_mode)
