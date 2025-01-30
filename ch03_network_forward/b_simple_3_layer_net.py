import numpy as np

from ch03_network_forward.a_activation_function import sigmoid, identity_function


class Simple3LayerNN:
    """Three-layer full connect neural network.

    Graphical representation of the network:

        Input Layer      Hidden Layer 1     Hidden Layer 2      Output Layer
            x1  --------> (signoid ) -------> (sigmoid) -------->(identify)->y1
                            /     \\           /     \\
            x2  --------> (sigmoid ) -------> (sigmoid) -------->(identify)->y2
                            \\     /
                        -> (sigmoid)

    Explanation:
    - Input Layer:
        - 2 Nodes: x1, x2
        - Fully connected to all nodes in Hidden Layer 1.
    - Hidden Layer 1:
        - 3 nodes, fully connected to Input Layer and Hidden Layer 2.
    - Hidden Layer 2:
        - 2 nodes, fully connected to Hidden Layer 1 and Output Layer.
    - Output Layer:
        - 2 nodes: y1, y2.
        - Fully connected to Hidden Layer 2.

    This network is fully connected and feedforward.
    """

    def __init__(
        self,
        init_param: dict[str, np.typing.NDArray[np.floating]] | None = None,
    ) -> None:
        """Initialize weights and biases."""
        if init_param is not None:
            self.parameters = init_param
        else:
            np.random.seed(0)  # For reproducibility
            self.parameters = {
                "W1": np.random.randn(2, 3),
                "b1": np.random.randn(3),
                "W2": np.random.randn(3, 2),
                "b2": np.random.randn(2),
                "W3": np.random.randn(2, 2),
                "b3": np.random.randn(2),
            }

    def forward(
        self, x: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """Forward pass of the network.

        The formulation of the network is as follows:
        - Hidden Layer 1: sigmoid(x * W1 + b1)
        - Hidden Layer 2: sigmoid(h1 * W2 + b2)
        - Output Layer: h2 * W3 + b3

        Parameters:
            x (np.typing.NDArray[np.floating]): Input array.

        Returns:
            np.typing.NDArray[np.floating] of float: Output array after forward pass.
        """
        y1 = sigmoid(np.dot(x, self.parameters["W1"]) + self.parameters["b1"])
        y2 = sigmoid(np.dot(y1, self.parameters["W2"]) + self.parameters["b2"])
        return identity_function(
            np.dot(y2, self.parameters["W3"]) + self.parameters["b3"]
        )
