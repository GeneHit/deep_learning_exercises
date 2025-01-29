from typing import Sequence, TypeAlias

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


def array(sequencia: Sequence[float]) -> NDArray[np.floating]:
    """Convert a list to a numpy array."""
    return np.array(sequencia, dtype=NN_FLOAT_TYPE)


def zeros(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a new array of given shape and type, filled with zeros."""
    return np.zeros(shape, dtype=NN_FLOAT_TYPE)


def ones(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a new array of given shape and type, filled with ones."""
    return np.ones(shape, dtype=NN_FLOAT_TYPE)


def randn(shape: tuple[int, ...]) -> NDArray[np.floating]:
    """Return a sample (or samples) from the "standard normal" distribution."""
    return np.random.randn(*shape).astype(NN_FLOAT_TYPE)


def normal(
    loc: float, scale: float, size: tuple[int, ...]
) -> NDArray[np.floating]:
    """Draw random samples from a normal (Gaussian) distribution."""
    return np.random.normal(loc, scale, size).astype(NN_FLOAT_TYPE)


def uniform(
    low: float, high: float, size: tuple[int, ...]
) -> NDArray[np.floating]:
    """Draw samples from a uniform distribution."""
    return np.random.uniform(low, high, size).astype(NN_FLOAT_TYPE)
