import numpy as np
import pytest

from ch04_network_learning.a_loss_function import (
    cross_entropy_error,
    mean_squared_error,
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
        (np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.0, 0.0]), 0.025),
        (
            np.array([0.0, 0.1, 0.2]),
            np.array([0.1, 0.1, 0.1]),
            0.01,
        ),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.0),
        (
            np.array([0.0, 4.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
            12.5,
        ),
        # 2-dimensional example
        (
            np.array([[0.0, 0.1], [0.2, 0.3]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            0.07,
        ),
        # 3-dimensional example
        (
            np.array([[[0.0, 0.1], [0.2, 0.3]], [[0.0, 0.0], [0.0, 0.0]]]),
            np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            0.07,
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
            np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]),
            np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            2.30258409,
        ),
        # 1-dimensional example
        (np.array([0.1, 0.9]), np.array([0, 1]), 0.105360516),
        (np.array([0.8, 0.2]), np.array([1, 0]), 0.223143551),
        (np.array([0.5, 0.5]), np.array([0, 1]), 0.693147181),
        # Example with input close to zero
        (
            np.array([1e-10, 1.0 - 1e-10]),
            np.array([0, 1]),
            1.000000082690371e-10,
        ),
        (
            np.array([1.0 - 1e-10, 1e-10]),
            np.array([1, 0]),
            1.000000082690371e-10,
        ),
        # # 2-dimensional example
        (
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            np.array([[0, 1], [1, 0]]),
            0.3285038,
        ),
        # # 3-dimensional example
        (
            np.array(
                [[[0.1, 0.9], [0.8, 0.2]], [[0.5, 0.5], [1e-10, 1.0 - 1e-10]]]
            ),
            np.array([[[0, 1], [1, 0]], [[0, 1], [0, 1]]]),
            1.0216507,
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
