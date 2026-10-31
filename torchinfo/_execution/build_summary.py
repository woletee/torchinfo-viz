from __future__ import annotations
from torchinfo._enums.run_settings import Verbosity
from torchinfo._execution.forward.forward_pass import forward_pass
from torchinfo._execution.setup.prepare_input import prepare_input
from torchinfo._execution.summary_args import SummaryArgs
from torchinfo._execution.setup.summary_options import summary_options
from torchinfo._inputs.memory.get_total_memory_used import get_total_memory_used
from torchinfo.formatting import FormattingOptions
from torchinfo.model_statistics import ModelStatistics


def build_summary(args: SummaryArgs) -> ModelStatistics:
    """Validate the arguments, run the model and build its statistics."""
    columns, rows, model_mode, verbose = summary_options(args)
    x, sizes, device = prepare_input(args, columns, verbose)
    cache = bool(args.cache_forward_pass)
    model, batch_dim, kwargs = args.model, args.batch_dim, args.kwargs
    layers = forward_pass(model, x, batch_dim, cache, device, model_mode, **kwargs)
    formatting = FormattingOptions(args.depth, verbose, columns, args.col_width, rows)
    results = ModelStatistics(layers, sizes, get_total_memory_used(x), formatting)
    if verbose > Verbosity.QUIET:
        print(results)
    return results
