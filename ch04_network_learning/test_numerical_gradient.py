from typing import Callable

import numpy as np
import pytest

from ch04_network_learning.c_numerical_gradient import (
    GradientWith1LayerNN,
    numerical_gradient,
    numerical_gradient_descend,
)

ATOL = 1e-5


@pytest.mark.parametrize(
    "func, init_x, expected_gradient",
    [
        # 1D examples
        (lambda x: np.sum(x**2), np.array([3.0, 4.0]), np.array([6.0, 8.0])),
        (lambda x: np.sum(x**2), np.array([1.0, -1.0]), np.array([2.0, -2.0])),
        # 2D examples
        (
            lambda x: np.sum(x**2),
            np.array([[1.0, -1.0], [2.0, 3.2]]),
            np.array([[2.0, -2.0], [4.0, 6.4]]),
        ),
        # 3D examples
        (
            lambda x: np.sum(x**2),
            np.array([[[1.0, -1.0], [2.0, 3.2]], [[3.4, 4.6], [1.9, 2.4]]]),
            np.array([[[2.0, -2.0], [4.0, 6.4]], [[6.8, 9.2], [3.8, 4.8]]]),
        ),
        # Sine and Cosine function with 2-d variable
        (
            lambda x: np.sin(x[0]) + np.cos(x[1]),
            np.array([np.pi / 4, np.pi / 3]),
            np.array([np.cos(np.pi / 4), -np.sin(np.pi / 3)]),
        ),
        # Exponential and Logarithmic function with 2-d variable
        (
            lambda x: np.exp(x[0]) + np.log(x[1]),
            np.array([1.0, 2.0]),
            np.array([np.exp(1.0), 1 / 2.0]),
        ),
        # Multiplication function with 2-d variable
        (lambda x: x[0] * x[1], np.array([2.0, 3.0]), np.array([3.0, 2.0])),
    ],
)
def test_numerical_gradient(
    func: Callable[[np.typing.NDArray[np.floating]], float],
    init_x: np.typing.NDArray[np.floating],
    expected_gradient: np.typing.NDArray[np.floating],
) -> None:
    result = numerical_gradient(func, init_x)
    assert np.allclose(result, expected_gradient, atol=ATOL)


@pytest.mark.parametrize(
    "func, init_x, lr, step_num, expected_minimum",
    [
        # TODO: decide a better step_num with the later implementation
        # 1D examples
        (
            lambda x: np.sum(x**2),
            np.array([3.0, 4.0]),
            0.1,
            100,
            np.array([0.0, 0.0]),
        ),
        # 2D examples
        (
            lambda x: np.sum(x**2),
            np.array([[1.0, -1.0]]),
            0.1,
            100,
            np.array([[0.0, 0.0]]),
        ),
        # 3D examples
        (
            lambda x: np.sum(x**2),
            np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            0.1,
            100,
            np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        ),
        (
            lambda x: np.sum(x**2),
            np.array([[[1.0, -1.0], [2.0, 3.2]], [[3.4, 4.6], [1.9, 2.4]]]),
            0.1,
            100,
            np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        ),
    ],
)
def test_numerical_gradient_descend(
    func: Callable[[np.typing.NDArray[np.floating]], float],
    init_x: np.typing.NDArray[np.floating],
    lr: float,
    step_num: int,
    expected_minimum: np.typing.NDArray[np.floating],
) -> None:
    result = numerical_gradient_descend(func, init_x, lr, step_num)
    assert np.allclose(result, expected_minimum, atol=ATOL)


@pytest.mark.parametrize(
    "param, x, t, expected_gradient",
    [
        (
            np.array(
                [
                    [0.47355232, 0.9977393, 0.84668094],
                    [0.85557411, 0.03563661, 0.69422093],
                ]
            ),
            np.array([0.6, 0.9]),
            np.array([0.0, 0.0, 1.0]),
            np.array(
                [
                    [0.21924763, 0.14356247, -0.36281009],
                    [0.32887144, 0.2153437, -0.54421514],
                ]
            ),
        )
        # TODO: add more test cases
    ],
)
def test_gradient_with_simple_nn(
    param: np.typing.NDArray[np.floating],
    x: np.typing.NDArray[np.floating],
    t: np.typing.NDArray[np.floating],
    expected_gradient: np.typing.NDArray[np.floating],
) -> None:
    nn = GradientWith1LayerNN(param)
    result = nn.gradient(x, t)
    assert np.allclose(result, expected_gradient, atol=ATOL)
