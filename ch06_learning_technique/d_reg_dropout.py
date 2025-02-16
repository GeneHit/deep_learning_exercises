import numpy as np
from numpy.typing import NDArray

from common.base import Layer
from common.default_type_array import np_float


class Dropout(Layer):
    """Dropout layer over element.

    During training, randomly set some of the input elements to zero.
    This is a kind of regularization technique.
    """

    def __init__(
        self, dropout_ratio: float = 0.5, inplace: bool = False
    ) -> None:
        assert 0 <= dropout_ratio < 1.0
        self._dropout_ratio = np_float(dropout_ratio)
        self._inplace = inplace
        self._training = False
        self._mask: NDArray[np.bool] | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train(self, flag: bool) -> None:
        """See the base class."""
        raise NotImplementedError

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the dropout layer.

        There are standard and scale-invert dropout.
        """
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the dropout layer."""
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        return {}


class Dropout2d(Layer):
    """Special dropout layer over channle for image.

    The input can be of shape (N, C, H, W), where N is the batch size,
    C is the number of channels, and H and W are the height and width.
    This is a kind of regularization technique.
    """

    def __init__(
        self, dropout_ratio: float = 0.5, inplace: bool = False
    ) -> None:
        """Initialize the dropout layer.

        Parameters:
            dropout_ratio : float
                The ratio of the elements to be set to zero.
            inplace : bool
                If True, the operation is performed inplace, meaning the
                input tensor is directly modified without allocating additional
                memory for the output tensor. This can save memory and improve
                computational efficiency, particularly for large models or
                inputs. However, use with caution as it modifies the input
                data directly, which may lead to unintended side effects
                if the input is reused elsewhere.
        """
        assert 0 <= dropout_ratio < 1.0
        self._dropout_ratio = np_float(dropout_ratio)
        self._inplace = inplace
        self._training = False
        self._mask: NDArray[np.bool] | None = None

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
