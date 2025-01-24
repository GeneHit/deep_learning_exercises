from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.base import NueralNet


@dataclass(frozen=True, kw_only=True)
class ConvParams:
    """Convolutional layer parameters."""

    filter_num: int
    filter_h: int
    filter_w: int
    pad: int
    stride: int


@dataclass(frozen=True, kw_only=True)
class PoolParams:
    """Pool layer parameters."""

    pool_h: int
    pool_w: int
    stride: int
    pad: int


@dataclass(frozen=True, kw_only=True)
class SimpleCNNParams:
    """Simple Convolutional Neural Network parameters.

    (conv - relu - max_pool) - (flatten - affine - relu) - affine - softmax
            layer1                  layer2(hidden)         output
    """

    input_dim: tuple[int, int, int]
    """The dimensions of the data, not including the batch size."""

    conv_params: ConvParams

    pooling_params: PoolParams

    hidden_size: int
    """The number of neurons in the hidden layer2."""

    output_size: int
    """The number of neurons in the output layer."""

    weight_init_std: float | None = None
    """The standard deviation of the weight initialization. If None, it will
    choose the Xavier or He initialization based on the activation inside
    """


class SimpleCNN(NueralNet):
    """Simple Convolutional Neural Network.

    (conv - relu - max_pool) - (flatten - affine - relu) - affine - softmax
            layer1                  layer2(hidden)         output
    """

    def __init__(self, params: SimpleCNNParams) -> None:
        self._params = params

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
