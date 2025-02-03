import numpy as np
from numpy.typing import NDArray

from common.base import Layer
from common.default_type_array import np_ones


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
        self._x: NDArray[np.floating] | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")

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
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")


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
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        d(affine)/dx = dout * W.T
        d(affine)/dW = x.T * dout
        d(affine)/db = sum(dout, axis=0)
        """
        raise NotImplementedError("The backward method is not implemented yet.")

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        raise NotImplementedError


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
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Softmax Backward Pass (Correct Jacobian-vector multiplication).

        dSoftmax(x)/dx = Softmax(x) * (dout - Sum(dout * Softmax(x), axis=-1))
        """
        raise NotImplementedError("The backward method is not implemented yet.")


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
        raise NotImplementedError("Do not need to implemente this method now.")

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
        raise NotImplementedError("The method is not implemented yet.")

    def backward(
        self, dout: NDArray[np.floating] = np_ones(shape=(1,))
    ) -> NDArray[np.floating]:
        """Backward pass of the layer.

        d(softmax_cross_entropy_loss)/dx = (y - t) / batch_size

        Parameters:
            dout : NDArray[np.floating]
                Gradient of the loss. Usually, it is 1.0 = dL/dL.
        """
        raise NotImplementedError("The backward method is not implemented yet.")
