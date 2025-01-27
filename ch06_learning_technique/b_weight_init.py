import numpy as np
from numpy.typing import NDArray


def generate_init_weight(
    weight_shape: tuple[int, ...],
    initializer: str = "he_normal",
    mode: str = "fan_in",
    stddev: float | None = None,
) -> NDArray[np.floating]:
    """Generate initialized weights.

    the current best practice:

          activation function next to               initialization method
        --------------------------------------------------------------------
                    ReLU                                He initialization
        sigmoid or tanh (S-shaped curve function)     Xavier initialization

    Affine layer: 2D array, with shape (D_in, D_out).
    Convolution layer: 4D array, with shape (F_n, C_in, K_h, K_w).

    -> fan_in = D_in or C_in * K_h * K_w
    -> fan_out = D_out or F_n * K_h * K_w
    scale = fan_in or fan_out or (fan_in + fan_out) / 2 (different modes)

    Normal distribution with mean 0 and variance:
        he: std = sqrt(2 / scale)
        xavier: std = sqrt(1 / scale).
    Uniform distribution:  sqrt(3) * above_std.

    Parameters:
        weight_shape : Tuple[int, ...]
            Shape of the weight tensor. only 2D or 4D.
        initializer : str
            The initializer for the weights of the layer.
            Options:
                - "he_normal": He normal initializer.
                - "he_uniform": He uniform initializer.
                - "xavier_normal": Xavier normal initializer.
                - "xavier_uniform": Xavier uniform initializer.
                - "normal": Normal distribution with weight_init_std.
                - "uniform": Uniform distribution with weight_init_std.
        mode : str
            The mode for the initialization. It is used for the He/Xavier
            initialization. Options: "fan_in", "fan_out", fan_out or fan_avg.
        stddev : float, optional
            Standard deviation for normal/uniform distribution. It is only used
            when the initializer is "normal" or "uniform".

    Returns:
        np.ndarray: Initialized weight tensor.
    """
    raise NotImplementedError


def generate_init_bias(
    bias_shape: tuple[int, ...],
    initializer: str = "zeros",
    stddev: float | None = None,
) -> NDArray[np.floating]:
    """Generate initialized biases.

    Parameters:
        bias_shape : Tuple[int, ...]
            Shape of the bias tensor.
        initializer : str
            The initializer for the biases of the layer.
            Options:
                - "zeros": Initialize all biases to 0.
                - "ones": Initialize all biases to 1.
                - "normal": Normal distribution with weight_init_std.
                - "uniform": Uniform distribution with weight_init_std.
        stddev : float, optional
            Standard deviation for normal/uniform distribution. It is only used
            when the initializer is "normal" or "uniform".

    Returns:
        np.ndarray: Initialized bias tensor.
    """
    raise NotImplementedError
