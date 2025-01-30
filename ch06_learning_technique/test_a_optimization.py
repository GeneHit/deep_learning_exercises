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
from common.default_type_array import get_default_type, np_array, np_float

ATOL = 1e-1


def df_for_test(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """The gradient of the function f(x) = x^2."""
    return np_float(2) * x


def _update_params(
    optimizer: Optimizer,
    init_pos: NDArray[np.floating],
    step: int,
) -> NDArray[np.floating]:
    # Update parameters using SGD
    pos = {"key": init_pos}
    for _ in range(step):
        optimizer.one_step(params=pos, grads={"key": df_for_test(init_pos)})
    return pos["key"]


@pytest.mark.parametrize(
    "init_pos, step, lr",
    [
        (np_array([-7.0, 2.0, 3.5]), 30, 0.1),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 30, 0.1),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 30, 0.1),
    ],
)
def test_sgd(
    init_pos: NDArray[np.floating],
    step: int,
    lr: float,
) -> None:
    # Initialize the optimizer
    optimizer = SGD(lr=lr)

    # Update parameters
    pos = _update_params(optimizer, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step, lr",
    [
        (np_array([-7.0, 2.0, 3.5]), 30, 0.1),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 30, 0.1),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 30, 0.1),
    ],
)
def test_momentum(
    init_pos: NDArray[np.floating],
    step: int,
    lr: float,
    momentum: float = 0.9,
) -> None:
    # Initialize the optimizer
    optimizer = Momentum(lr, momentum)

    # Update parameters
    pos = _update_params(optimizer, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step, lr",
    [
        (np_array([-7.0, 2.0, 3.5]), 30, 0.1),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 30, 0.1),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 30, 0.1),
    ],
)
def test_ada_grad(
    init_pos: NDArray[np.floating],
    step: int,
    lr: float,
) -> None:
    # Initialize the optimizer
    optimizer = AdaGrad(lr)

    # Update parameters using SGD
    pos = _update_params(optimizer, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step, lr",
    [
        (np_array([-7.0, 2.0, 3.5]), 30, 0.1),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 30, 0.1),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 30, 0.1),
    ],
)
def test_rms_prop(
    init_pos: NDArray[np.floating],
    step: int,
    lr: float,
    decay_rate: float = 0.9,
) -> None:
    # Initialize the optimizer
    optimizer = RMSProp(lr, decay_rate=decay_rate)

    # Update parameters
    pos = _update_params(optimizer, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)


@pytest.mark.parametrize(
    "init_pos, step, lr",
    [
        (np_array([-7.0, 2.0, 3.5]), 30, 0.1),
        (np_array([[-7.0, 2.0], [5.5, -4.0]]), 30, 0.1),
        (np_array([[[-7.0], [2.0]], [[5.5], [-4.0]]]), 30, 0.1),
    ],
)
def test_adam(
    init_pos: NDArray[np.floating],
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> None:
    # Initialize the optimizer
    optimizer = Adam(lr, beta1, beta2)

    # Update parameters
    pos = _update_params(optimizer, init_pos, step)
    assert pos.dtype == get_default_type()

    # Check if the updated parameters are close to the origin
    assert np.allclose(pos, np.zeros_like(pos), atol=ATOL)
