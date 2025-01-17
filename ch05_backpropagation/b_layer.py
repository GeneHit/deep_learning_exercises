import abc
import numpy as np
from numpy.typing import NDArray


class LayerBase(abc.ABC):
    """Base class for neural network layers."""

    @abc.abstractmethod
    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """

    def train_flag(self, flag: bool) -> None:
        """Set the training flag of the layer.

        During training, some layer may need to change their behavior, for
        example, dropout layer.

        Parameters:
            flag : bool
                Training flag. During training, the flag cann be set to True
                if neccesary.
        """
        pass

    @abc.abstractmethod
    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Parameters:
            x (NDArray[np.floating]): Input data.

        Returns:
            NDArray[np.floating]: Output data.
        """

    @abc.abstractmethod
    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout (NDArray[np.floating]): Gradient of the loss.

        Returns:
            NDArray[np.floating]: Gradient of the input, given to next layer.
        """

    @abc.abstractmethod
    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the network.

        This have to be called after the backward pass.
        """


class ReLU(LayerBase):
    """Layer that applies the Rectified Linear Unit (ReLU) activation function.

    The graphical representation of the layer is:

        x  ----> ReLU: max(0, x) ----> y
    """

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


class Sigmoid(LayerBase):
    """Layer that applies the Sigmoid activation function.

    The graphical representation of the layer is:

        x ----> Sigmoid: 1 / (1 + exp(-x)) ----> y
    """

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


class Affine(LayerBase):
    """Layer that performs the affine transformation.

    The graphical representation of the layer is:

        x  ----> W ----> xW + b ----> y
    """

    def __init__(
        self, W: tuple[str, NDArray[np.floating]], b: tuple[str, NDArray[np.floating]]
    ) -> None:
        self._W_name, self._W = W
        self._b_name, self._b = b

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {self._W_name: self._W, self._b_name: self._b}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        raise NotImplementedError


class SoftmaxWithLoss(LayerBase):
    """Layer that calculates the softmax and cross-entropy loss.

    The graphical representation of the layer is:

        x ----> Softmax: exp(x) / sum(exp(x)) ----> y ----> CrossEntropyLoss
                                                    t -----/
    """

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError("Do not need to implemente this method.")

    def forward_to_loss(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        """Calculate the loss."""
        raise NotImplementedError("The loss method is not implemented yet.")

    def backward(
        self, dout: NDArray[np.floating] = np.array([1.0])
    ) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        return {}
