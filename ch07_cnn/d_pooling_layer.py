import numpy as np
from numpy.typing import NDArray

from common.base import Layer


class MaxPool2d(Layer):
    """Max pooling layer.

    Pooling operations, whether 2D or 3D, are applied independently to
    each channel of the input data. This means that the pooling operation
    processes the spatial dimensions (height and width for 2D pooling,
    depth, height, and width for 3D pooling) but does not mix or combine
    information across different channels.
    """

    def __init__(
        self, kenel_size: tuple[int, int], stride: int = 2, pad: int = 0
    ) -> None:
        """Initialize the MaxPool2d layer.

        Parameters:
            kenel_size: tuple[int, int]
                The size of the pooling window. The tuple should have two
                integers: (height, width).
            stride: int
                The stride of the pooling operation.
            pad: int
                The padding applied to the input data.
        """
        self._kenel_size = kenel_size
        self._stride = stride
        self._pad = pad

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        # There are no parameters to update in the MaxPool2d layer
        return {}


class AvgPool2d(Layer):
    """Average pooling layer.

    Pooling operations, whether 2D or 3D, are applied independently to
    each channel of the input data. This means that the pooling operation
    processes the spatial dimensions (height and width for 2D pooling,
    depth, height, and width for 3D pooling) but does not mix or combine
    information across different channels.
    """

    def __init__(
        self, kenel_size: tuple[int, int], stride: int = 2, pad: int = 0
    ) -> None:
        self._kenel_size = kenel_size
        self._stride = stride
        self._pad = pad

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        # There are no parameters to update in the AvgPool2d layer
        return {}


class Flatten(Layer):
    """Flatten layer.

    This operation is crucial for transitioning from convolutional layers to
    fully connected layers. The Flatten layer reshapes the input data into a
    2D array, with shape (batch_size, n).
    How to calculate n:
        n = channel * height * width
    """

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).

        Returns:
            NDArray[np.floating]: The reshaped array, with shape (batch_size, n).
        """
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 2D array:
                    (batch_size, n).

        Returns:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        # There are no parameters to update in the Flatten layer
        return {}
