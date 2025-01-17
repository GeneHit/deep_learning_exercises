import pytest
import numpy as np
from ch03_network_forward.b_simple_3_layer_net import Simple3LayerNN


@pytest.mark.parametrize(
    "params, x, expected_output",
    [
        # Example from the book
        (
            {
                "W1": np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
                "b1": np.array([0.1, 0.2, 0.3]),
                "W2": np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
                "b2": np.array([0.1, 0.2]),
                "W3": np.array([[0.1, 0.3], [0.2, 0.4]]),
                "b3": np.array([0.1, 0.2]),
            },
            np.array([1.0, 0.5]),
            np.array([0.31682708, 0.69627909]),
        ),
    ],
)
def test_forward(
    params: dict[str, np.typing.NDArray[np.floating]],
    x: np.typing.NDArray[np.floating],
    expected_output: np.typing.NDArray[np.floating],
) -> None:
    # Create the network
    net: Simple3LayerNN = Simple3LayerNN(init_param=params)

    # Perform the forward pass
    output: np.typing.NDArray[np.floating] = net.forward(x)

    # Assert the output is as expected
    assert np.allclose(output, expected_output, atol=1e-4)
