import pytest
import numpy as np

from ch04_network_learning.a_loss_function import (
    mean_squared_error,
    cross_entropy_error,
)

ATOL = 1e-6


@pytest.mark.parametrize(
    "y, t, expected",
    [
        # Example from the book
        (
            np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]),
            np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            0.0975,
        ),
        # 1-dimensional example
        (np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2]), 0.0),
        (np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.0, 0.0]), 0.01),
        (np.array([0.0, 0.1, 0.2]), np.array([0.1, 0.1, 0.1]), 0.0033333333333333335),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.0),
        (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), 4.666666666666667),
        # 2-dimensional example
        (np.array([[0.0, 0.1], [0.2, 0.3]]), np.array([[0.0, 0.0], [0.0, 0.0]]), 0.025),
        # 3-dimensional example
        (
            np.array([[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]]]),
            np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            0.1225,
        ),
    ],
)
def test_mean_squared_error(
    y: np.typing.NDArray[np.floating],
    t: np.typing.NDArray[np.floating],
    expected: float,
) -> None:
    assert np.isclose(mean_squared_error(y, t), expected, atol=ATOL)


@pytest.mark.parametrize(
    "y, t, expected",
    [
        # Example from the book
        (
            np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]),
            2.3025,
        ),
        # 1-dimensional example
        (np.array([0.1, 0.9]), np.array([0, 1]), 0.105360516),
        (np.array([0.8, 0.2]), np.array([1, 0]), 0.223143551),
        (np.array([0.5, 0.5]), np.array([0, 1]), 0.693147181),
        # Example with input close to zero
        (np.array([1e-10, 1.0 - 1e-10]), np.array([0, 1]), 1.000000082690371e-10),
        (np.array([1.0 - 1e-10, 1e-10]), np.array([1, 0]), 1.000000082690371e-10),
        # 2-dimensional example
        (np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([[0, 1], [1, 0]]), 0.164252033),
        # 3-dimensional example
        (
            np.array([[[0.1, 0.9], [0.8, 0.2]], [[0.5, 0.5], [1e-10, 1.0 - 1e-10]]]),
            np.array([[[0, 1], [1, 0]], [[0, 1], [0, 1]]]),
            0.346573591,
        ),
    ],
)
def test_cross_entropy_error(
    y: np.typing.NDArray[np.floating],
    t: np.typing.NDArray[np.floating],
    expected: float,
) -> None:
    output = cross_entropy_error(y, t)
    assert np.isclose(output, expected, atol=ATOL)
