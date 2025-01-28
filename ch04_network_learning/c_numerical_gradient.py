from typing import Callable

import numpy as np


def numerical_gradient(
    f: Callable[[np.typing.NDArray[np.floating]], float],
    x: np.typing.NDArray[np.floating],
) -> np.typing.NDArray[np.floating]:
    """Calculate numerical gradient for a n-dimensional function.

    The formulation of numerical gradient is as follows:
        grad = [(f(x + h) - f(x - h)) / (2 * h)]_i

    Notice: the x is a mutable:
        1. if you want to keep the original x0, you should copy it before the
        calculation.
        2. if you want to change the x0, you can directly use it.

    Parameters:
        f : Callable[[np.typing.NDArray[np.floating]], float])
            Function to differentiate.
        x : np.typing.NDArray[np.floating])
            Point to differentiate.

    Returns:
        np.typing.NDArray[np.floating]: Numerical gradient.
    """
    raise NotImplementedError


def numerical_gradient_descend(
    f: Callable[[np.typing.NDArray[np.floating]], float],
    x: np.typing.NDArray[np.floating],
    learning_rate: float,
    step_num: int,
) -> np.typing.NDArray[np.floating]:
    """Calculate numerical gradient descend for a n-dimensional function.

    The formulation of numerical gradient descend is as follows:
        x = x - learning_rate * grad

    Parameters:
        f : Callable[[np.typing.NDArray[np.floating]], float])
            Function to differentiate.
        x : np.typing.NDArray[np.floating])
            Point to differentiate.
        learning_rate : float
            Learning rate for gradient descend.
        step_num : int
            Number of steps to descend.

    Returns:
        np.typing.NDArray[np.floating]: Numerical gradient descend.
    """
    raise NotImplementedError


class GradientWith1LayerNN:
    """Gradient with simple neural network.

    The graph of the simple neural network is as follows:
        x -> [W] -> [softmax] -> y
    where x is 2D array.

    We use a simple neural network to demonstrate the gradient calculation.
    """

    def __init__(self, param: np.typing.NDArray[np.floating]) -> None:
        self._param = param

    def forward(
        self, x: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:
        """Forward the pass of the simple neural network.

        Parameters:
            x : np.typing.NDArray[np.floating]
                Input data.

        Returns:
            np.typing.NDArray[np.floating]: Output data.
        """
        raise NotImplementedError

    def loss(
        self,
        x: np.typing.NDArray[np.floating],
        t: np.typing.NDArray[np.floating],
    ) -> float:
        """Calculate loss of the simple neural network.

        Parameters:
            x : np.typing.NDArray[np.floating]
                Input data.
            t : np.typing.NDArray[np.floating]
                Target data.

        Returns:
            float: Loss.
        """
        raise NotImplementedError

    def gradient(
        self,
        x: np.typing.NDArray[np.floating],
        t: np.typing.NDArray[np.floating],
    ) -> np.typing.NDArray[np.floating]:
        """Calculate gradient of the simple neural network.

        Parameters:
            x : np.typing.NDArray[np.floating]
                Input data.
            t : np.typing.NDArray[np.floating]
                Target data.
        """
        raise NotImplementedError
