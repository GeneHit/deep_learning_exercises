import numpy as np
from numpy.typing import NDArray

from ch04_network_learning.d_learning_implementation import TwoLayerNN
from ch05_backpropagation.b_layer import Affine, ReLU, SoftmaxWithLoss
from common.base import Layer
from common.default_type_array import np_float, np_randn, np_zeros


class BackPropTwoLayerNN(TwoLayerNN):
    """A 2-layer neunal network.

    Graphical representation of the network:

    x ---affine---ReLU---affine---softmax---y----> CrossEntropyLoss
                                          t -----/

    This network is fully connected and feedforward, using 2D-array data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        init_std = np_float(weight_init_std)
        self._params = {
            "W1": init_std * np_randn((input_size, hidden_size)),
            "b1": np_zeros((1, hidden_size)),
            "W2": init_std * np_randn((hidden_size, output_size)),
            "b2": np_zeros((1, output_size)),
        }
        self._layers: tuple[Layer, ...] = (
            Affine(("W1", self._params["W1"]), ("b1", self._params["b1"])),
            ReLU(inplace=True),
            Affine(("W2", self._params["W2"]), ("b2", self._params["b2"])),
        )
        self._last_layer = SoftmaxWithLoss()

    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters by outside +=, -=.
        """
        return self._params

    def predict(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def loss(self, x: NDArray[np.floating], t: NDArray[np.floating]) -> float:
        y = self.predict(x)
        return self._last_layer.forward_to_loss(y, t)

    def accuracy(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            # one-hot encoding, like [[0, 0, 1, 0, 0], ]
            t = np.argmax(t, axis=1)
        return float(np.sum(y == t) / x.shape[0])

    def gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        # forward
        self.loss(x, t)

        # backward
        dout = self._last_layer.backward()  # dout's default is 1.0
        for layer in reversed(self._layers):
            dout = layer.backward(dout)

        # collect gradients
        grads: dict[str, NDArray[np.floating]] = {}
        for layer in self._layers:
            grads.update(layer.param_grads())
        return grads
