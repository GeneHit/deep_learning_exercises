import numpy as np
from numpy.typing import NDArray

from common.base import Layer
from common.default_type_array import np_float


class BatchNorm1d(Layer):
    """Batch normalization layer over feactue.

    The input can either be of shape (N, C) or (N, C, L), where
    N is the batch size, C is the number of features, and L is
    the sequence length for 3D inputs. Be used for fully connected layers, RNNs.

    Normalizes over features (C)
        bn = BatchNorm1d(num_features=10)
        * Input shape: (batch_size, num_features)
        * output = bn(randn(32, 10)) (with 2D array)
        - Input shape: (batch_size, num_features, sequence_length)
        - output = bn(randn(32, 10, 20)) (with 3D array)

    Paper: http://arxiv.org/abs/1502.03167
    The formulation is as follows:

        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta
    """

    def __init__(
        self,
        gamma: tuple[str, NDArray[np.floating]],
        beta: tuple[str, NDArray[np.floating]],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """
        Custom implementation of Batch Normalization.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A small constant added to the denominator for numerical stability.
            momentum (float): Momentum for updating running statistics.
            affine (bool): If True, learnable affine parameters (gamma and beta) are used.
            track_running_stats (bool): If True, running mean and variance are tracked during training.
        """
        # num_features = gamma[1].size
        assert gamma[1].ndim == 1 and beta[1].ndim == 1
        self._params = {gamma[0]: gamma[1], beta[0]: beta[1]}
        self._eps = np_float(eps)
        self._momentum = np_float(momentum)
        self._affine = affine
        self._track_running_stats = track_running_stats

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        if self._affine:
            return self._params
        return {}

    def train(self, flag: bool) -> None:
        """See the base class."""
        raise NotImplementedError

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        raise NotImplementedError


class BatchNorm2d(Layer):
    """Batch normalization layer over channel.

    The input can either be of shape (N, C, H, W), where
    N is the batch size, C is the number of channels, and H and W are
    the height and width of the input. Be used for Image data (2D convolutions).

    Paper: http://arxiv.org/abs/1502.03167
    The formulation is as follows:

        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta
    """

    def __init__(
        self,
        gamma: tuple[str, NDArray[np.floating]],
        beta: tuple[str, NDArray[np.floating]],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """
        Custom implementation of Batch Normalization.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A small constant added to the denominator for numerical stability.
            momentum (float): Momentum for updating running statistics.
            affine (bool): If True, learnable affine parameters (gamma and beta) are used.
            track_running_stats (bool): If True, running mean and variance are tracked during training.
        """
        # num_features = gamma[1].size
        assert gamma[1].ndim == 1 and beta[1].ndim == 1
        self._params = {gamma[0]: gamma[1], beta[0]: beta[1]}
        self._eps = np_float(eps)
        self._momentum = np_float(momentum)
        self._affine = affine
        self._track_running_stats = track_running_stats

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return self._params

    def train(self, flag: bool) -> None:
        """See the base class."""
        raise NotImplementedError

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        raise NotImplementedError
