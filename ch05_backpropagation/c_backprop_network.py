import numpy as np
from numpy.typing import NDArray

from common.base import NueralNet


class TwoLayerNN(NueralNet):
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
        raise NotImplementedError

    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters by outside +=, -=.
        """
        raise NotImplementedError

    def predict(
        self, x: NDArray[np.floating], train_flag: bool = False
    ) -> NDArray[np.floating]:
        raise NotImplementedError

    def loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        train_flag: bool = False,
    ) -> float:
        raise NotImplementedError

    def accuracy(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        raise NotImplementedError

    def gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        raise NotImplementedError
