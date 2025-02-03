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
    return float(0.5 * np.sum((y - t) ** 2))


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
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    return float(-np.sum(t * np.log(y + 1e-7)))


def cross_entropy_error_single_lable(
    y: np.typing.NDArray[np.floating],
    t: np.typing.NDArray[np.floating | np.integer],
) -> float:
    """Calculate cross entropy error for lable.

    The formulation of cross entropy error is as follows:
        error = -sum(t * log(y))

    Parameters:
        y (np.typing.NDArray[np.floating | np.integer]): Predicted array.
        t (np.typing.NDArray[np.floating]): Target array.

    Returns:
        float: Cross entropy error.
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # If the target data is one-hot-vector, convert it to
    # the index of the correct label
    if t.size == y.size:
        t = t.argmax(axis=1)

    assert np.issubdtype(t.dtype, np.integer), (
        f"t.dtype={t.dtype}, expected integer"
    )
    batch_size = y.shape[0]
    return float(
        -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / float(batch_size)
    )
