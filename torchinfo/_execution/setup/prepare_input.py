from __future__ import annotations
from torchinfo._execution.setup.validate_user_params import validate_user_params
from torchinfo._inputs.device.get_device import get_device
from torchinfo._inputs.process_input import process_input


def prepare_input(args, columns, verbose):
    """Validate the user's arguments and build the model's sample input."""
    input_data, input_size, dtypes = args.input_data, args.input_size, args.dtypes
    device, input_device = args.device, None
    if input_size is not None:
        device = device if device is not None else get_device(args.model, None)
        input_device = device
    validate_user_params(
        input_data, input_size, columns, args.col_width, device, dtypes, verbose
    )
    x, sizes = process_input(
        input_data, input_size, args.batch_dim, input_device, dtypes
    )
    return (x, sizes, input_device)
