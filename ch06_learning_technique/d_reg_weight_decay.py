import numpy as np
from numpy.typing import NDArray

from common.base import NueralNet


class MultiLinearNN(NueralNet):
    """Multi-layer neural network with linear affine.

    Graphical representation of the network:

        Input Layer                Hidden 1     ...         Output Layer
            x ---affine->(activation)-->        ...--->(affine)-->y
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        activation: str = "relu",
        weight_init_std: float | None = None,
        weight_decay_lambda: float = 0.0,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        dropout_ratio: float = 0.5,
    ) -> None:
        """Initialize the MultiLayerNN.

        Parameters:
            input_size (int): Input size.
            hidden_sizes (tuple[int, ...]): Hidden layer sizes.
            output_size (int): Output size.
            activation (str): Activation function name.
            weight_init_std (float, optional): Standard deviation of the weight initialization.
            weight_decay_lambda (float): Weight decay factor.
            use_batchnorm (bool): Flag to use batch normalization.
            use_dropout (bool): Flag to use dropout.
            dropout_ratio (float): Dropout ratio.
        """
        self._input_size = input_size
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size
        self._activation = activation
        self._weight_init_std = weight_init_std
        self._weight_decay_lambda = weight_decay_lambda
        self._use_batchnorm = use_batchnorm
        self._use_dropout = use_dropout
        self._dropout_ratio = dropout_ratio

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
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def gradient(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
    ) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        raise NotImplementedError
