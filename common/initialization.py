import numpy as np
from numpy.typing import NDArray

NN_FLOAT_TYPE = np.float32


def compute_fan_in_and_fan_out(
    weight_shape: tuple[int, ...],
) -> tuple[int, int]:
    """Compute fan_in and fan_out based on the weight shape."""
    match len(weight_shape):
        case 2:  # Affine layer
            fan_in, fan_out = weight_shape[0], weight_shape[1]
        case 4:  # Convolution layer
            fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3]
            fan_out = weight_shape[0] * weight_shape[2] * weight_shape[3]
        case _:
            raise ValueError("The weight shape must be 2D or 4D.")
    return fan_in, fan_out


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
    fan_in, fan_out = compute_fan_in_and_fan_out(weight_shape)

    # Determine scaling factor based on mode using match-case
    scale = 0.0  # for mypy
    match mode:
        case "fan_in":
            scale = fan_in
        case "fan_out":
            scale = fan_out
        case "fan_avg":
            scale = (fan_in + fan_out) / 2
        case _:
            raise ValueError("Mode must be 'fan_in', 'fan_out', or 'fan_avg'.")

    # Initialize weights based on the chosen initializer using match-case
    match initializer:
        case "he_normal":
            std = np.sqrt(2.0 / scale)
            return np.random.normal(
                loc=0.0, scale=std, size=weight_shape
            ).astype(NN_FLOAT_TYPE)

        case "he_uniform":
            limit = np.sqrt(6.0 / scale)
            return np.random.uniform(
                low=-limit, high=limit, size=weight_shape
            ).astype(NN_FLOAT_TYPE)

        case "xavier_normal":
            std = np.sqrt(1.0 / scale)
            return np.random.normal(
                loc=0.0, scale=std, size=weight_shape
            ).astype(NN_FLOAT_TYPE)

        case "xavier_uniform":
            limit = np.sqrt(3.0 / scale)
            return np.random.uniform(
                low=-limit, high=limit, size=weight_shape
            ).astype(NN_FLOAT_TYPE)

        case "normal":
            assert stddev is not None, (
                "stddev must be specified for 'normal' initializer."
            )
            return np.random.normal(
                loc=0.0, scale=stddev, size=weight_shape
            ).astype(NN_FLOAT_TYPE)

        case "uniform":
            assert stddev is not None, (
                "stddev must be specified for 'uniform' initializer."
            )
            limit = np.sqrt(3.0) * stddev
            return np.random.uniform(
                low=-limit, high=limit, size=weight_shape
            ).astype(NN_FLOAT_TYPE)

        case _:
            raise ValueError(
                "Initializer must be one of 'he_normal', 'he_uniform', "
                "'xavier_normal', 'xavier_uniform', 'normal', or 'uniform'."
            )


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
    match initializer:
        case "zeros":
            return np.zeros(bias_shape, dtype=NN_FLOAT_TYPE)

        case "ones":
            return np.ones(bias_shape, dtype=NN_FLOAT_TYPE)

        case "normal":
            assert stddev is not None, (
                "stddev must be specified for 'normal' initializer."
            )
            return np.random.normal(
                loc=0.0, scale=stddev, size=bias_shape
            ).astype(NN_FLOAT_TYPE)

        case "uniform":
            assert stddev is not None, (
                "stddev must be specified for 'uniform' initializer."
            )
            limit = np.sqrt(3.0) * stddev
            return np.random.uniform(
                low=-limit, high=limit, size=bias_shape
            ).astype(NN_FLOAT_TYPE)

        case _:
            raise ValueError(
                "Initializer must be one of 'zeros', 'ones', 'normal', or 'uniform'."
            )
