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
        running_mean: tuple[str, NDArray[np.floating]],
        running_var: tuple[str, NDArray[np.floating]],
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
        self._gamma_name = gamma[0]
        self._beta_name = beta[0]
        self._running_mean_name = running_mean[0]
        self._running_var_name = running_var[0]
        self._params = {
            gamma[0]: gamma[1],
            beta[0]: beta[1],
            running_mean[0]: running_mean[1],
            running_var[0]: running_var[1],
        }
        self._eps = np_float(eps)
        self._momentum = np_float(momentum)
        self._affine = affine
        self._track_running_stats = track_running_stats

    def __post_init__(self) -> None:
        gamma = self._params[self._gamma_name]
        assert gamma.ndim == 2
        num_features = gamma.shape[1]
        assert (
            (1, num_features)
            == gamma.shape
            == self._params[self._beta_name].shape
            == self._params[self._running_mean_name].shape
            == self._params[self._running_var_name].shape
        )

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
        """Backward pass of the layer.

        out = gamma * x_hat + beta
        N = batch_size
        dL/d_gamma:
            = dL/d_out * dout/d_gamma = sum( dout * x_hat.T , axis=0)
        dL/d_beta:
            = dL/d_out * dout/d_beta = sum( dout, axis=0)
        dL/d_x_hat:
            = dL/d_out * dout/d_x_hat = dout * gamma
        dL/d_var:
            = dL/d_x_hat * dx_hat/d_var
            = sum( dL/d_x_hat * (x - mean) * -0.5 * (var + eps)^-1.5, axis=0)
        dL/d_mean:
            = dL/d_x_hat * dx_hat/d_mean
            = (
                sum( d_x_hat * -1 / sqrt(var + eps), axis=0) +
                dL/d_var * sum(-2 * (x - mean), axis=0) / N
            )
        dL/d_x:
            = dL/d_x_hat * dx_hat/d_x
            = (
                dL/d_x_hat / sqrt(var + eps) +
                dL/d_var * 2 * (x - mean) / N + dL/d_mean / N
            )
        """
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
        running_mean: tuple[str, NDArray[np.floating]],
        running_var: tuple[str, NDArray[np.floating]],
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
        # num_channel = gamma[1].size
        self._gamma_name = gamma[0]
        self._beta_name = beta[0]
        self._running_mean_name = running_mean[0]
        self._running_var_name = running_var[0]
        self._params = {
            gamma[0]: gamma[1],
            beta[0]: beta[1],
            running_mean[0]: running_mean[1],
            running_var[0]: running_var[1],
        }
        self._eps = np_float(eps)
        self._momentum = np_float(momentum)
        self._affine = affine
        self._track_running_stats = track_running_stats

    def __post_init__(self) -> None:
        gamma = self._params[self._gamma_name]
        assert gamma.ndim == 4
        num_channel = gamma.shape[1]
        assert (
            (1, num_channel, 1, 1)
            == gamma.shape
            == self._params[self._beta_name].shape
            == self._params[self._running_mean_name].shape
            == self._params[self._running_var_name].shape
        )

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
        """Backward pass of the layer.

        out = gamma * x_hat + beta
        N = batch_size * H * W
        dL/d_gamma:
            = dL/d_out * dout/d_gamma = sum( dout * x_hat.T , axis=(0, 2, 3))
        dL/d_beta:
            = dL/d_out * dout/d_beta = sum( dout, axis=(0, 2, 3))
        dL/d_x_hat:
            = dL/d_out * dout/d_x_hat = dout * gamma
        dL/d_var:
            = dL/d_x_hat * dx_hat/d_var
            = sum( dL/d_x_hat * (x - mean) * -0.5 * (var + eps)^-1.5, axis=(0, 2, 3))
        dL/d_mean:
            = dL/d_x_hat * dx_hat/d_mean
            = (
                sum( d_x_hat * -1 / sqrt(var + eps), axis=(0, 2, 3)) +
                dL/d_var * sum(-2 * (x - mean), axis=(0, 2, 3)) / N
            )
        dL/d_x:
            = dL/d_x_hat * dx_hat/d_x
            = (
                dL/d_x_hat / sqrt(var + eps) +
                dL/d_var * 2 * (x - mean) / N + dL/d_mean / N
            )
        """
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        raise NotImplementedError
