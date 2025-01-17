import numpy as np
from typing import Callable


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
    raise NotImplementedError


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
    raise NotImplementedError
