import numpy as np


def mean_squared_error(
    y: np.typing.NDArray[np.floating], t: np.typing.NDArray[np.floating]
) -> float:
    """Calculate mean squared error.

    The formulation of mean squared error is as follows:
        error = 1/2 * sum((y - t)^2)

    Parameters:
        y (np.typing.NDArray[np.floating]): Predicted array.
        t (np.typing.NDArray[np.floating]): Target array.

    Returns:
        float: Mean squared error.
    """
    raise NotImplementedError


def cross_entropy_error(
    y: np.typing.NDArray[np.floating], t: np.typing.NDArray[np.floating]
) -> float:
    """Calculate cross entropy error.

    The formulation of cross entropy error is as follows:
        error = -sum(t * log(y))

    Parameters:
        y (np.typing.NDArray[np.floating]): Predicted array.
        t (np.typing.NDArray[np.floating]): Target array.

    Returns:
        float: Cross entropy error.
    """
    raise NotImplementedError
