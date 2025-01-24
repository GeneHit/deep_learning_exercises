import numpy as np
from numpy.typing import NDArray


def generate_init_weight(
    shape: tuple[int, ...], method: str = "he", stddev: float | None = None
) -> NDArray[np.floating]:
    """Initialize weights for a network layer.

    the current best practice:

          activation function next to               initialization method
        --------------------------------------------------------------------
                    ReLU                                He initialization
        sigmoid or tanh (S-shaped curve function)     Xavier initialization

    Parameters:
        shape : tuple[int, ...])
            Shape of the weight matrix. The first element should be the
            number of last-layer neurons.
        method : str
            Initialization method. Default is 'he'.
        stddev : float)
            Standard deviation for random initialization. If
            provided, it overrides the method.

    Returns:
        NDArray[np.floating]: Initialized weights.
    """
    raise NotImplementedError
