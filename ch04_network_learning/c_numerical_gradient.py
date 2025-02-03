from typing import Callable

import numpy as np

from ch03_network_forward.a_activation_function import softmax
from ch04_network_learning.a_loss_function import cross_entropy_error

H = 1e-4


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
    original_value = 0.0
    grad = np.zeros_like(x)

    # using np.nditer for n-dimensional array iteration, avoiding nested loops
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])  # type: ignore
    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]
        # f(x + h)
        x[idx] = original_value + H
        f_x_plus_h = f(x)
        # f(x - h)
        x[idx] = original_value - H
        f_x_minus_h = f(x)
        # reset x[idx] to original value
        x[idx] = original_value
        # calculate gradient
        grad[idx] = (f_x_plus_h - f_x_minus_h) / (2 * H)
        it.iternext()
    return grad


def numerical_gradient_descend(
    f: Callable[[np.typing.NDArray[np.floating]], float],
    x: np.typing.NDArray[np.floating],
    learning_rate: float,
    step_num: int,
) -> np.typing.NDArray[np.floating]:
    """Calculate numerical gradient descend for a n-dimensional function.

    The formulation of stochastic gradient descend is as follows:
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
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= learning_rate * grad
    return x


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
        return softmax(np.dot(x, self._param))

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
        y = self.forward(x)
        return cross_entropy_error(y, t)

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

        # the self.loss already has the self._param. unsed is for mypy
        def loss_w(unused: np.typing.NDArray[np.floating]) -> float:
            return self.loss(x, t)

        # the numerical_gradient will change the self._param inside, since the
        # self._param is mutable.
        return numerical_gradient(loss_w, self._param)
