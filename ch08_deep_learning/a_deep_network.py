import pickle

import numpy as np
from numpy.typing import NDArray

from common.base import Layer


class Deep2dNet(Layer):
    """Deep 2D Neural Network.

    diagram:
        input - hidden_layers - ouput
    """

    def __init__(self, layers: tuple[Layer, ...]) -> None:
        self._layers = layers

        raise NotImplementedError

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters by outside -=, +=.
        """
        raise NotImplementedError

    def train(self, flag: bool) -> None:
        """Set the training flag for the network."""
        for layer in self._layers:
            layer.train(flag)

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """See the base class."""
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        raise NotImplementedError

    def save_params(self, file_name: str = "deep_2d_net_params.pkl") -> None:
        params = self.named_params()
        with open(file_name, "wb") as f:
            pickle.dump(params, f)
