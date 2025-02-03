from typing import Callable

import numpy as np
import pytest

from ch04_network_learning.b_numerical_diffirentiation import (
    numerical_diff_1d,
    numerical_partial_diff,
)

ATOL = 1e-4


@pytest.mark.parametrize(
    "f, x, expected",
    [
        (lambda x: 3 * x + 2, 5.0, 3.0),  # Linear function
        (lambda x: x**2, 3.0, 6.0),  # Quadratic function
        (lambda x: np.sin(x), np.pi / 4, np.cos(np.pi / 4)),  # Sine function
        (lambda x: np.exp(x), 1.0, np.exp(1.0)),  # Exponential function
        (lambda x: np.log(x), 2.0, 1 / 2.0),  # Logarithmic function
        (lambda x: x**3, 2.0, 12.0),  # Cubic function
        (lambda x: np.cos(x), np.pi / 3, -np.sin(np.pi / 3)),  # Cosine function
        (
            lambda x: np.tan(x),
            np.pi / 6,
            1 / (np.cos(np.pi / 6) ** 2),
        ),  # Tangent function
        (lambda x: 1 / x, 1.0, -1.0),  # Reciprocal function
        (
            lambda x: np.sqrt(x),
            4.0,
            1 / (2 * np.sqrt(4.0)),
        ),  # Square root function
    ],
)
def test_numerical_diff_1d(
    f: Callable[[float], float], x: float, expected: float
) -> None:
    result = numerical_diff_1d(f, x)
    assert np.isclose(result, expected, atol=ATOL)


@pytest.mark.parametrize(
    "f, x, axis, expected",
    [
        # linear function with 1-d variable
        (lambda x: np.sum(x), np.array([1.0, 2.0]), (1,), 1.0),
        # Quadratic function with 1-d variable
        (lambda x: np.sum(x**2), np.array([3.0, 4.0]), (1,), 8.0),
        # Cubic function with 1-d variable
        (lambda x: np.sum(x**3), np.array([2.0, 3.0]), (0,), 12.0),
        # Quadratic function with 2-d variable
        (
            lambda x: np.sum(x**2),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            (0, 1),
            4.0,
        ),
        # Multiplication function with 1-d variable
        (lambda x: x[0] * x[1], np.array([2.0, 3.0]), (0,), 3.0),
        # Sine and Cosine function with 1-d variable
        (
            lambda x: np.sin(x[0]) + np.cos(x[1]),
            np.array([np.pi / 4, np.pi / 3]),
            (0,),
            np.cos(np.pi / 4),
        ),
        # Exponential and Logarithmic function with 1-d variable
        (
            lambda x: np.exp(x[0]) + np.log(x[1]),
            np.array([1.0, 2.0]),
            (0,),
            np.exp(1.0),
        ),
        # Tangent and Reciprocal function with 1-d variable
        (
            lambda x: np.tan(x[0]) + 1 / x[1],
            np.array([np.pi / 6, 1.0]),
            (1,),
            -1.0,
        ),
        # Square root function with 1-d variable
        (
            lambda x: np.sqrt(x[0]) + x[1] ** 0.5,
            np.array([4.0, 9.0]),
            (0,),
            1 / (2 * np.sqrt(4.0)),
        ),
    ],
)
def test_numerical_partial_diff(
    f: Callable[[np.typing.NDArray[np.floating]], float],
    x: np.typing.NDArray[np.floating],
    axis: tuple[int, ...],
    expected: float,
) -> None:
    result = numerical_partial_diff(f=f, x0=x, axis=axis)
    assert np.allclose(result, expected, atol=ATOL)
