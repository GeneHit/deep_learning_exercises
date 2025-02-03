import numpy as np
from numpy.typing import NDArray

from ch04_network_learning.a_loss_function import (
    cross_entropy_error_single_lable,
)
from common.base import Layer
from common.default_type_array import np_float, np_ones


class ReLU(Layer):
    """Layer that applies the Rectified Linear Unit (ReLU) activation function.

    The graphical representation of the layer is:

        x  ----> ReLU: max(0, x) ----> y
    """

    def __init__(self, inplace: bool = False) -> None:
        """Initialize the layer.

        Parameters:
            inplace : bool
                If True, the operation is performed inplace, meaning the
                input tensor is directly modified without allocating additional
                memory for the output tensor. This can save memory and improve
                computational efficiency, particularly for large models or
                inputs. However, use with caution as it modifies the input
                data directly, which may lead to unintended side effects
                if the input is reused elsewhere.
        """
        self._inplace = inplace
        # self._mask: NDArray[np.bool_] | None = None
        self._x: NDArray[np.floating] | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        self._x = x

        return np.maximum(0, x, out=x if self._inplace else None, dtype=x.dtype)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        assert self._x is not None
        result = dout * (self._x > 0)
        self._x = None  # free memory
        return result

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        return {}


class Sigmoid(Layer):
    """Layer that applies the Sigmoid activation function.

    The graphical representation of the layer is:

        x ----> Sigmoid: 1 / (1 + exp(-x)) ----> y
    """

    def __init__(self, inplace: bool = False) -> None:
        self._inplace = inplace
        self._y: NDArray[np.floating] | None = None

    # This layer have no parameters learned by backward gradient.
    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        # when inplace = True, self._y references to x
        self._y = np_float(1) / (
            np_float(1) + np.exp(-x, out=x if self._inplace else None)
        )
        assert self._y is not None  # for mypy
        return self._y

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        dSigmoid(x)/dx = Sigmoid(x) * (1 - Sigmoid(x))
        """
        assert self._y is not None
        result: NDArray[np.floating] = dout * self._y * (np_float(1) - self._y)
        self._y = None  # free memory
        return result


class Affine(Layer):
    """Layer that performs the affine transformation.

    Only for 2D input and 2D weight matrix.

    The graphical representation of the layer is:
        x  ----> xW + b ----> y
    """

    def __init__(
        self,
        w: tuple[str, NDArray[np.floating]],
        b: tuple[str, NDArray[np.floating]],
    ) -> None:
        self._w_name, self._w = w
        self._b_name, self._b = b

        self._x: NDArray[np.floating] | None = None
        self._dx: NDArray[np.floating] | None = None
        self._dw: NDArray[np.floating] | None = None
        self._db: NDArray[np.floating] | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {self._w_name: self._w, self._b_name: self._b}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        self._x = x
        result: NDArray[np.floating] = np.dot(x, self._w) + self._b  # for mypy
        return result

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        d(affine)/dx = dout * W.T
        d(affine)/dW = x.T * dout
        d(affine)/db = sum(dout, axis=0)
        """
        assert self._x is not None
        dx: NDArray[np.floating] = np.dot(dout, self._w.T)
        self._dw = np.dot(self._x.T, dout)
        self._db = np.sum(dout, axis=0)
        self._x = None  # free memory
        return dx

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        assert self._dw is not None and self._db is not None
        return {self._w_name: self._dw, self._b_name: self._db}


class Softmax(Layer):
    """Layer that calculates the softmax.

    The graphical representation of the layer is:

        x ----> Softmax: exp(x) / sum(exp(x)) ----> y
    """

    def __init__(self, inplace: bool = False) -> None:
        self._inplace = inplace
        self._y: NDArray[np.floating] | None = None

    # This layer have no parameters learned by backward gradient.
    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Softmax Forward Pass"""
        # minus max(x) for numerical stability
        if self._inplace:
            x -= np.max(x, axis=-1, keepdims=True)
            # buffer references to x
            buffer = x
        else:
            # create a new buffer
            buffer = x - np.max(x, axis=-1, keepdims=True)

        np.exp(buffer, out=buffer)
        # Normalize along last axis
        sum_exp_x = np.sum(buffer, axis=-1, keepdims=True)
        # exp(x) / sum(exp(x))
        buffer /= sum_exp_x
        self._y = buffer
        assert self._y is not None  # for mypy
        return self._y

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Softmax Backward Pass (Correct Jacobian-vector multiplication).

        dSoftmax(x)/dx = Softmax(x) * (dout - Sum(dout * Softmax(x), axis=-1))
        """
        assert self._y is not None
        sum_dout_y = np.sum(dout * self._y, axis=-1, keepdims=True)
        result: NDArray[np.floating] = self._y * (dout - sum_dout_y)  # for mypy
        self._y = None  # free memory
        return result


class SoftmaxWithLoss(Layer):
    """Layer that calculates the softmax and cross-entropy loss.

    Only for 2D input and 2d/1D target.

    The graphical representation of the layer is:

        x ----> Softmax: exp(x) / sum(exp(x)) ----> y ----> CrossEntropyLoss
                                                    t -----/
    """

    def __init__(self) -> None:
        # this layer always is the last layer in the network, we can write
        # to the input x.
        self._softmax = Softmax(inplace=True)
        self._y: NDArray[np.floating] | None = None
        self._t: NDArray[np.floating | np.integer] | None = None

    # This layer have no parameters learned by backward gradient.
    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        self._y = self._softmax.forward(x)
        return self._y

    def forward_to_loss(
        self, x: NDArray[np.floating], t: NDArray[np.floating | np.integer]
    ) -> float:
        """Forward pass of the layer.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            float: Loss value.
        """
        self._y = self.forward(x)
        self._t = t
        return cross_entropy_error_single_lable(self._y, t)

    def backward(
        self, dout: NDArray[np.floating] = np_ones(shape=(1,))
    ) -> NDArray[np.floating]:
        """Backward pass of the layer.

        d(softmax_cross_entropy_loss)/dx = (y - t) / batch_size

        Parameters:
            dout : NDArray[np.floating]
                Gradient of the loss. Usually, it is 1.0 = dL/dL.
        """
        assert self._y is not None
        assert self._t is not None
        batch_size = self._t.shape[0]
        if self._t.size == self._y.size:
            # the lable is one-hot encoded, like [0, 0, 1, 0, 0]
            dx = (self._y - self._t.astype(self._y.dtype)) / np_float(
                batch_size
            )
        else:
            dx = self._y
            dx[np.arange(batch_size), self._t] -= 1
            dx /= np_float(batch_size)

        self._y = None  # free memory
        self._t = None  # free memory
        return dx
