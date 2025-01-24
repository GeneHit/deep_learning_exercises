import numpy as np
from numpy.typing import NDArray


def generate_init_weight(
    layer_type: str,
    weight_shape: tuple[int, ...],
    activation: str = "relu",
    distribution: str = "normal",
    stddev: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate initialized weights for a given layer.

    the current best practice:

          activation function next to               initialization method
        --------------------------------------------------------------------
                    ReLU                                He initialization
        sigmoid or tanh (S-shaped curve function)     Xavier initialization

    He initialization:
        Normal distribution with mean 0 and variance (
            2/D_in or 2/(C_in * K_h * K_w)
        )
    Xavier initialization:
        Normal distribution with mean 0 and variance (
            2/(D_in + D_out) or 2/(C_in * K_h * K_w + F_n * K_h * K_w)
        ). But, usually, it is variance 1/D_in or 1/(C_in * K_h * K_w).

    If requeiring a uniform distribution, have to multi 3 inside the variance.

    Parameters:
        layer_type : str
            Type of layer ("conv" or "affine").
        weight_shape : Tuple[int, ...])
            Shape of the weight tensor.
                - For Conv layers: (F_n, C_in, K_h, K_w)
                - For Affine layers: (D_in, D_out)
        activation : str
            Activation function type ("relu", "sigmoid", "tanh", or "softmax").
        distribution : str
            Distribution type for initialization ("normal" or "uniform").
        stddev : float, optional
            Standard deviation for normal distribution. If provided,
            it will use this value instead use the He/Xavier method.

    Returns:
        np.ndarray: Initialized weight tensor.
    """
    raise NotImplementedError
