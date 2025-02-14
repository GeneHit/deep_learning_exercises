import numpy as np
from numpy.typing import NDArray

from common.base import Layer
from common.default_type_array import get_default_type, np_float, np_sqrt


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

        self._dgamma: NDArray[np.floating] | None = None
        self._dbeta: NDArray[np.floating] | None = None
        self._training = True

        # buffer for forward pass, avoiding recalculation
        self._divide_sqrt_var: NDArray[np.floating] | None = None
        self._x_hat: NDArray[np.floating] | None = None
        self._x_minus_mean: NDArray[np.floating] | None = None

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
        return self._params

    def train(self, flag: bool) -> None:
        """See the base class."""
        self._training = flag

    def _process_mean_var(
        self, x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Calculate the mean and variance of the input.

        Returns:
            the batch mean : NDArray[np.floating]
        """
        # (BatchSize, N) -> (1, N)
        batch_mean: NDArray[np.floating] = np.mean(x, axis=0, keepdims=True)
        batch_var = np.var(x, axis=0, keepdims=True)
        self._divide_sqrt_var = np_float(1) / np_sqrt(batch_var + self._eps)

        if self._track_running_stats:
            self._params[self._running_mean_name] = (
                self._momentum * self._params[self._running_mean_name]
                + (np_float(1) - self._momentum) * batch_mean
            )
            self._params[self._running_var_name] = (
                self._momentum * self._params[self._running_var_name]
                + (np_float(1) - self._momentum) * batch_var
            )

        return batch_mean

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        if self._training or not self._track_running_stats:
            batch_mean = self._process_mean_var(x)
            assert self._divide_sqrt_var is not None

            self._x_minus_mean = x - batch_mean
            x_hat: NDArray[np.floating] = (
                self._x_minus_mean * self._divide_sqrt_var
            )
            self._x_hat = x_hat
        else:
            running_mean = self._params[self._running_mean_name]
            running_var = self._params[self._running_var_name]
            x_hat = (x - running_mean) / np_sqrt(running_var + self._eps)

        if not self._affine:
            return x_hat

        gamma = self._params[self._gamma_name]
        beta = self._params[self._beta_name]
        result: NDArray[np.floating] = gamma * x_hat + beta
        return result

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
        assert self._x_hat is not None
        if self._affine:
            self._dgamma = np.sum(dout * self._x_hat, axis=0)
            self._dbeta = np.sum(dout, axis=0)
            d_x_hat = dout * self._params[self._gamma_name]
        else:
            d_x_hat = dout

        assert self._x_minus_mean is not None
        assert self._divide_sqrt_var is not None
        d_x_hat_dot_divide_sqrt_var = d_x_hat * self._divide_sqrt_var
        batch_size = self._x_hat.shape[0]
        d_var = np.sum(
            (
                d_x_hat_dot_divide_sqrt_var
                * self._x_minus_mean
                * -0.5
                * (self._divide_sqrt_var**2)
            ),
            axis=0,
            dtype=get_default_type(),
        )
        d_mean = (
            np.sum(-d_x_hat_dot_divide_sqrt_var, axis=0)
            + d_var * np.sum(-2 * self._x_minus_mean, axis=0) / batch_size
        ).astype(get_default_type())
        d_x: NDArray[np.floating] = (
            d_x_hat_dot_divide_sqrt_var
            + d_var * np_float(2) * self._x_minus_mean / batch_size
            + d_mean / d_x_hat.size
        ).astype(get_default_type())
        # clear the buffer
        self._x_hat = None
        self._x_minus_mean = None
        self._divide_sqrt_var = None
        return d_x

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        if not self._affine:
            return {}

        assert self._dgamma is not None and self._dbeta is not None
        return {
            self._gamma_name: self._dgamma,
            self._beta_name: self._dbeta,
        }


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

        self._dgamma: NDArray[np.floating] | None = None
        self._dbeta: NDArray[np.floating] | None = None
        self._training = True

        # buffer for forward pass, avoiding recalculation
        self._divide_sqrt_var: NDArray[np.floating] | None = None
        self._x_hat: NDArray[np.floating] | None = None
        self._x_minus_mean: NDArray[np.floating] | None = None

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
        self._training = flag

    def _process_mean_var(
        self, x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Calculate the mean and variance of the input.

        Returns:
            the batch mean : NDArray[np.floating]
        """
        # (N, C, H, W) -> (1, C, 1, 1)
        batch_mean: NDArray[np.floating] = np.mean(
            x, axis=(0, 2, 3), keepdims=True
        )
        batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
        self._divide_sqrt_var = np_float(1) / np_sqrt(batch_var + self._eps)

        if self._track_running_stats:
            self._params[self._running_mean_name] = (
                self._momentum * self._params[self._running_mean_name]
                + (np_float(1) - self._momentum) * batch_mean
            )
            self._params[self._running_var_name] = (
                self._momentum * self._params[self._running_var_name]
                + (np_float(1) - self._momentum) * batch_var
            )

        return batch_mean

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        running_mean = self._params[self._running_mean_name]
        running_var = self._params[self._running_var_name]
        if self._training or not self._track_running_stats:
            batch_mean = self._process_mean_var(x)

            assert self._divide_sqrt_var is not None
            self._x_minus_mean = x - batch_mean
            x_hat: NDArray[np.floating] = (
                self._x_minus_mean * self._divide_sqrt_var
            )
            self._x_hat = x_hat
        else:
            x_hat = (x - running_mean) / np_sqrt(running_var + self._eps)

        if not self._affine:
            return x_hat
        result: NDArray[np.floating] = (
            self._params[self._gamma_name] * x_hat
            + self._params[self._beta_name]
        )
        return result

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
        assert self._x_hat is not None
        if self._affine:
            self._dgamma = np.sum(
                dout * self._x_hat, axis=(0, 2, 3), keepdims=True
            )
            self._dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
            d_x_hat = dout * self._params[self._gamma_name]
        else:
            d_x_hat = dout

        assert self._x_minus_mean is not None
        assert self._divide_sqrt_var is not None
        d_x_hat_dot_divide_sqrt_var = d_x_hat * self._divide_sqrt_var
        _, c, h, w = self._x_hat.shape
        n = c * h * w
        d_var = np.sum(
            (
                d_x_hat_dot_divide_sqrt_var
                * self._x_minus_mean
                * -0.5
                * (self._divide_sqrt_var**2)
            ),
            axis=(0, 2, 3),
            dtype=get_default_type(),
        )
        d_mean = (
            np.sum(-d_x_hat_dot_divide_sqrt_var, axis=(0, 2, 3))
            + d_var
            * np.sum(-2 * self._x_minus_mean, axis=(0, 2, 3))
            / d_x_hat.size
        ).astype(get_default_type())
        d_var = d_var.reshape(1, -1, 1, 1)
        d_mean = d_mean.reshape(1, -1, 1, 1)
        d_x: NDArray[np.floating] = (
            d_x_hat_dot_divide_sqrt_var
            + d_var * np_float(2) * self._x_minus_mean / n
            + d_mean / n
        ).astype(get_default_type())
        return d_x

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        if not self._affine:
            return {}

        assert self._dgamma is not None and self._dbeta is not None
        return {
            self._gamma_name: self._dgamma,
            self._beta_name: self._dbeta,
        }
