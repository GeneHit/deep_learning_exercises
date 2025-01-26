import pickle
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.base import LayerConfig, NueralNet


@dataclass(frozen=True, kw_only=True)
class DeepConvNetConfig:
    """Simple Convolutional Neural Network parameters.

    diagram:
        conv - relu - conv- relu - max_pool -
        conv - relu - conv- relu - max_pool -
        conv - relu - conv- relu - max_pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    input_dim: tuple[int, int, int]
    """The dimensions of the data, not including the batch size."""

    hidden_layer_configs: tuple[LayerConfig, ...]
    """The configurations of the hidden layers."""

    hidden_size: int
    """The number of neurons in the hidden layer2."""

    output_size: int
    """The number of neurons in the output layer."""

    weight_init_std: float | None = None
    """The standard deviation of the weight initialization. If None, it will
    choose the Xavier or He initialization based on the activation inside
    """


class DeepConvNet(NueralNet):
    """Deep Convolutional Neural Network.

    diagram:
        conv - relu - conv- relu - max_pool -
        conv - relu - conv- relu - max_pool -
        conv - relu - conv- relu - max_pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, config: DeepConvNetConfig) -> None:
        self._config = config
        # just used for loading the existed parameters
        self._params: dict[str, NDArray[np.floating]] | None = {}

        raise NotImplementedError

    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """
        raise NotImplementedError

    def predict(
        self, x: NDArray[np.floating], train_flag: bool = False
    ) -> NDArray[np.floating]:
        """See the base class."""
        raise NotImplementedError

    def loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        train_flag: bool = False,
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def accuracy(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        raise NotImplementedError

    def save_params(self, file_name: str = "params.pkl") -> None:
        params = self.named_parameters()
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name: str = "params.pkl") -> None:
        with open(file_name, "rb") as f:
            params = pickle.load(f)

        if self._params is None:
            # get the mutable parameters of every layer, meaning that it gets
            # the reference of the parameters and can update the parameters.
            self._params = self.named_parameters()

        for key, val in params.items():
            self._params[key] = val
