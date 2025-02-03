import time
from contextlib import contextmanager
from typing import Iterator

from common.base import Layer
from common.default_type_array import get_default_type


@contextmanager
def log_duration(process: str) -> Iterator[None]:
    """Log duration of some processes.

    Parameters
    ----------
    process : str
        Describe the process with lowercase letters, numbers, or underscores.

    Raises
    ------
    ValueError
        If `process` is empty.
    """
    if process == "":
        raise ValueError("The process description cannot be empty.")

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(
            f"{process}_duration_s: {end_time - start_time:.1f}. "
            f"start_time_s: {start_time:.4f}; end_time_s: {end_time:.4f}."
        )


def assert_layer_parameter_type(layer: Layer) -> None:
    """Assert the type of the parameters in the layer.

    The parameters should have the same type as the default type. We can use
    this to verify that computations are done with the correct type.
    """
    for param_name, param_value in layer.named_params().items():
        assert param_value.dtype == get_default_type(), (
            f"Parameter {param_name} in the layer has the wrong type. "
            f"Expected {get_default_type()}, but got {param_value.dtype}."
        )

    for param_name, param_grad in layer.param_grads().items():
        assert param_grad.dtype == get_default_type(), (
            f"Parameter gradient {param_name} in the layer has the wrong type. "
            f"Expected {get_default_type()}, but got {param_grad.dtype}."
        )
