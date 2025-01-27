import numpy as np
from numpy.typing import NDArray

from common.base import Layer


class Dropout2d(Layer):
    """Dropout layer.

    During training, randomly set some of the input elements to zero.
    This is a kind of regularization technique.
    """

    def __init__(self, dropout_ratio: float = 0.5) -> None:
        """
        Custom implementation of Dropout.

        Args:
            dropout_ratio (float): Dropout ratio. The probability of setting an element to zero.
        """
        self._dropout_ratio = dropout_ratio

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train(self, flag: bool) -> None:
        """See the base class."""
        raise NotImplementedError

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the dropout layer."""
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the dropout layer."""
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        return {}
