import numpy as np


class Simple3LayerNN:
    """3-layer full connect neural network.

    Graphical representation of the network:

        Input Layer      Hidden Layer 1     Hidden Layer 2      Output Layer
            x  --------> (signoid ) -------> (sigmoid) -------->(softmax)->y

    This network is fully connected and feedforward.
    """

    def __init__(
        self, init_param: dict[str, np.typing.NDArray[np.floating]]
    ) -> None:
        """Initialize weights and biases."""
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            assert key in init_param, (
                f"{key} is not in the parameter dictionary."
            )
        self._parameters = init_param

    def _predict(
        self, x: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """Calculate the final output of neural network.

        The formulation of the network is as follows:
        - Hidden Layer 1: sigmoid(x * W1 + b1)
        - Hidden Layer 2: sigmoid(h1 * W2 + b2)
        - Output Layer: softmax(h2 * W3 + b3)

        Parameters:
            x (np.typing.NDArray[np.floating]): Input array.

        Returns:
            np.typing.NDArray[np.floating]: Output array after forward pass.
        """
        raise NotImplementedError

    def accuracy_with_for_cycle(
        self,
        x: np.typing.NDArray[np.floating],
        t: np.typing.NDArray[np.floating],
    ) -> float:
        """Calculate accuracy of the network.

        ** You have to implement this method using for loop (one by one data). **

        Parameters:
            x (np.typing.NDArray[np.floating]): Input array, 2D array now.
            t (np.typing.NDArray[np.floating]): Target array, 2D array now.

        Returns:
            float: Accuracy of the network.
        """
        raise NotImplementedError

    def accuracy_with_batch(
        self,
        x: np.typing.NDArray[np.floating],
        t: np.typing.NDArray[np.floating],
    ) -> float:
        """Calculate accuracy of the network.

        ** You have to implement this method without using for loop (one by one batch). **

        Parameters:
            x (np.typing.NDArray[np.floating]): Input array, 2D array now.
            t (np.typing.NDArray[np.floating]): Target array, 2D array now.

        Returns:
            float: Accuracy of the network.
        """
        raise NotImplementedError
