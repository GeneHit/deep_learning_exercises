from common.base import Layer
from common.default_type_array import get_default_type


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
