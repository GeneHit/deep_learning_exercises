from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ch06_learning_technique.a_optimization import (
    SGD,
    AdaGrad,
    Adam,
    Momentum,
    RMSProp,
)
from common.base import Optimizer
from common.default_type_array import (
    get_default_type,
    np_array,
    np_float,
    np_zeros_like,
)

ATOL = 1e-1


def gradient_for_sgd(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """The gradient of the function f(x) = x^2."""
    return np_float(2) * x


def df_for_test(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """The gradient of the function f(x) = x[:half]^2/20 + x[half:]^2."""
    half_size = x.size // 2
    x_flat = x.flatten()
    x_flat[:half_size] /= 10
    x_flat[half_size:] *= 2

    return x_flat.reshape(x.shape)


def _update_params(
    optimizer: Optimizer,
    grad_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    init_pos: NDArray[np.floating],
    step: int,
) -> NDArray[np.floating]:
    # Update parameters
    pos = {"key": init_pos}
    for _ in range(step):
        optimizer.one_step(params=pos, grads={"key": grad_func(pos["key"])})
    return pos["key"]


@pytest.mark.parametrize(
    "init_pos, step",
    [
        (np_array([-7.0, 2.0, 3.5]), 10),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 10),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 10),
    ],
)
def test_sgd(init_pos: NDArray[np.floating], step: int) -> None:
    # Initialize the optimizer
    optimizer = SGD(lr=0.3)

    # Update parameters
    pos = _update_params(optimizer, gradient_for_sgd, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step",
    [
        (np_array([-7.0, 2.0, 3.5]), 90),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 90),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 90),
    ],
)
def test_momentum(init_pos: NDArray[np.floating], step: int) -> None:
    # Initialize the optimizer
    optimizer = Momentum(lr=0.1, beta=0.9)

    # Update parameters
    pos = _update_params(optimizer, df_for_test, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np_zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step",
    [
        (np_array([-7.0, 2.0, 3.5]), 40),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 40),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 40),
    ],
)
def test_ada_grad(init_pos: NDArray[np.floating], step: int) -> None:
    # Initialize the optimizer
    optimizer = AdaGrad(lr=1.5)

    # Update parameters using
    pos = _update_params(optimizer, gradient_for_sgd, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np_zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step",
    [
        (np_array([-7.0, 2.0, 3.5]), 10),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 10),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 10),
    ],
)
def test_rms_prop(
    init_pos: NDArray[np.floating],
    step: int,
) -> None:
    # Initialize the optimizer
    optimizer = RMSProp(lr=1.5, decay_rate=0.99)

    # Update parameters
    pos = _update_params(optimizer, df_for_test, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step",
    [
        (np_array([-7.0, 2.0, 3.5]), 20),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 20),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 20),
    ],
)
def test_adam(init_pos: NDArray[np.floating], step: int) -> None:
    # Initialize the optimizer
    optimizer = Adam(lr=1.5, beta1=0.1, beta2=0.9)

    # Update parameters
    pos = _update_params(optimizer, df_for_test, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np_zeros_like(pos), atol=ATOL)
