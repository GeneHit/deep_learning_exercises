import numpy as np
from numpy.typing import NDArray


class TwoLayerNN:
    """A two-layer neural network.

    Graphical representation of the network:

        Input Layer      Hidden Layer 1      Output Layer
            x  --------> (signoid ) -------->(softmax)->y

    This network is fully connected and feedforward, which means using 2D array.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        self._params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """
        return self._params

    def predict(
        self,
        x: NDArray[np.floating],
        train_flag: bool = False,
    ) -> NDArray[np.floating]:
        """Predict the output for the given input.

        Parameters:
            x (NDArray[np.floating]): Input data.
            train_flag (bool): Training flag.

        Returns:
            NDArray[np.floating]: Predicted output.
        """
        raise NotImplementedError("The predict method is not implemented yet.")

    def loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        train_flag: bool = False,
    ) -> float:
        """Calculate the loss for the given input and target output.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.
            train_flag (bool): Training flag.

        Returns:
            float: Loss value.
        """
        raise NotImplementedError("The loss method is not implemented yet.")

    def accuracy(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        """Calculate the accuracy for the given input and target output.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            float: Accuracy value.
        """
        raise NotImplementedError("The accuracy method is not implemented yet.")

    def _numerical_gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        """Calculate the numerical gradient.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            dict[str, NDArray[np.floating]]: Gradients of the weights and biases.
        """
        raise NotImplementedError(
            "The numerical_gradient method is not implemented yet."
        )

    def gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        """Calculate the numerical gradient.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            dict[str, NDArray[np.floating]]: Gradients of the weights and biases.
        """
        return self._numerical_gradient(x, t)


def training(
    network: TwoLayerNN,
    x_train: NDArray[np.floating],
    t_train: NDArray[np.floating],
    learning_rate: np.floating,
    batch_size: int,
    epochs: int,
    verbose: bool = False,
) -> list[float]:
    """Train the network using the given training data.

    Now, you can use the stochastic (numerical) gradient descent method for
    training. And have to use the mini-batch method for training.

    Parameters:
        network (TwoLayerNN): Neural network to train.
        x_train (NDArray[np.floating]): Input data for training.
        t_train (NDArray[np.floating]): Target output for training.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        verbose (bool): Flag to print the training loss every epoch.

    Returns:
        training losses : tuple[list[float], list[float], list[float]]"""
    raise NotImplementedError("The training method is not implemented yet.")
