import numpy as np
from numpy.typing import NDArray

from ch03_network_forward.a_activation_function import sigmoid, softmax
from ch04_network_learning.a_loss_function import cross_entropy_error
from ch04_network_learning.c_numerical_gradient import numerical_gradient


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

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict the output for the given input.

        Parameters:
            x (NDArray[np.floating]): Input data.

        Returns:
            NDArray[np.floating]: Predicted output.
        """
        z1 = sigmoid(np.dot(x, self._params["W1"]) + self._params["b1"])
        return softmax(np.dot(z1, self._params["W2"]) + self._params["b2"])

    def loss(self, x: NDArray[np.floating], t: NDArray[np.floating]) -> float:
        """Calculate the loss for the given input and target output.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            float: Loss value.
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

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
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # one hot means the label is one hot encoded like [0, 0, 1, 0, 0]
        is_one_hot = t.ndim != 1
        target = np.argmax(t, axis=1) if is_one_hot else t
        return float(np.sum(y == target) / float(x.shape[0]))

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

        # the self.loss already has the self._param. unsed is for mypy
        def loss_w(unused: np.typing.NDArray[np.floating]) -> float:
            return self.loss(x, t)

        grad: dict[str, NDArray[np.floating]] = {}
        for key in self._params.keys():
            # the numerical_gradient will change the mutable self._param[key]
            # inside
            grad[key] = numerical_gradient(loss_w, self._params[key])

        return grad

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
        training losses : list[float]
    """
    training_losses: list[float] = []
    data_size = x_train.shape[0]
    batch_idx = 0
    params = network.named_parameters()
    for epoch in range(epochs):
        for i in range(0, data_size, batch_size):
            batch_idx += 1
            x_batch = x_train[i : i + batch_size]
            t_batch = t_train[i : i + batch_size]

            grad = network.gradient(x_batch, t_batch)
            for key in grad.keys():
                params[key] -= learning_rate * grad[key]

        loss = network.loss(x_train, t_train)
        training_losses.append(loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: loss {loss}")

    return training_losses
