from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NN_FLOAT_TYPE: TypeAlias = np.float32


def get_default_type() -> TypeAlias:
    """Return the default float type for neural networks."""
    return NN_FLOAT_TYPE


def set_default_type(float_type: TypeAlias) -> None:
    """Set the default float type for neural networks."""
    global NN_FLOAT_TYPE
    NN_FLOAT_TYPE = float_type


def np_array(mat) -> NDArray[np.floating]:  # type: ignore
    """Convert a list to a numpy array."""
    return np.array(mat, dtype=NN_FLOAT_TYPE)


def np_empty(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a empty array of given shape and type, without initialization."""
    return np.empty(shape=shape, dtype=NN_FLOAT_TYPE)


def np_float(value: float) -> NN_FLOAT_TYPE:
    """Return a floating-point number."""
    return NN_FLOAT_TYPE(value)


def np_normal(
    loc: float, scale: float, size: tuple[int, ...]
) -> NDArray[np.floating]:
    """Draw random samples from a normal (Gaussian) distribution."""
    return np.random.normal(loc, scale, size).astype(NN_FLOAT_TYPE)


def np_ones(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a new array of given shape and type, filled with ones."""
    return np.ones(shape, dtype=NN_FLOAT_TYPE)


def np_rand(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return random floats in the half-open interval [0.0, 1.0)."""
    return np.random.rand(*shape).astype(NN_FLOAT_TYPE)


def np_randn(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a sample (or samples) from the "standard normal" distribution."""
    return np.random.randn(*shape).astype(NN_FLOAT_TYPE)


def np_sqrt(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the non-negative square-root of an array, element-wise."""
    result = np.zeros_like(x, dtype=NN_FLOAT_TYPE)
    np.sqrt(x, out=result, dtype=NN_FLOAT_TYPE)
    return result


def np_uniform(
    low: float, high: float, size: tuple[int, ...]
) -> NDArray[np.floating]:
    """Draw samples from a uniform distribution."""
    return np.random.uniform(low, high, size).astype(NN_FLOAT_TYPE)


def np_zeros(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a new array of given shape and type, filled with zeros."""
    return np.zeros(shape, dtype=NN_FLOAT_TYPE)


def np_zeros_like(a: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return an array of zeros with the same shape and type as a given array."""
    return np.zeros_like(a, dtype=NN_FLOAT_TYPE)
