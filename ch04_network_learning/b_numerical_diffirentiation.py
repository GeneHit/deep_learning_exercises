from typing import Callable

import numpy as np

H = 1e-4


def numerical_diff_1d(f: Callable[[float], float], x0: float) -> float:
    """Calculate numerical differentiation for a 1-dimensional function.

    The formulation of numerical differentiation is as follows:
        diff = (f(x0 + h) - f(x0 - h)) / (2 * h)

    Parameters:
        f (callable): Function to differentiate.
        x0 (float): Point to differentiate.

    Returns:
        float: Numerical differentiation.
    """
    return (f(x0 + H) - f(x0 - H)) / (2 * H)


def numerical_partial_diff(
    f: Callable[[np.typing.NDArray[np.floating]], float],
    x0: np.typing.NDArray[np.floating],
    axis: tuple[int, ...],
) -> float:
    """Calculate numerical partial differentiation for a n-dimensional function.

    The formulation of numerical differentiation is as follows:
        diff = (f(x0 + h) - f(x0 - h)) / (2 * h)

    Parameters:
        f : Callable[[np.typing.NDArray[np.floating]], float])
            Function to differentiate.
        x0 : np.typing.NDArray[np.floating])
            Point to differentiate.
        axis : tuple[int, ...]
            Axis to differentiate, which represents the position of the variable of x.

    Returns:
        np.typing.NDArray[np.floating]: Numerical differentiation.
    """
    origin_value = x0[axis]
    # f(x0 - h)
    x0[axis] -= H
    f_x_minus_h = f(x0)
    # f(x0 + h)
    x0[axis] = origin_value + H
    f_x_plus_h = f(x0)
    # reset x0[axis] to original value
    x0[axis] = origin_value
    return (f_x_plus_h - f_x_minus_h) / (2 * H)
