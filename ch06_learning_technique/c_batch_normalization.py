import numpy as np
from numpy.typing import NDArray

from common.base import Layer


class BatchNorm2d(Layer):
    """Batch normalization layer.

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
        self._params = {gamma[0]: gamma[1], beta[0]: beta[1]}
        self._eps = eps
        self._momentum = momentum
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
