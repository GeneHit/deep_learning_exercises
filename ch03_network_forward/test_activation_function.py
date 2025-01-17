import pytest
import numpy as np
from ch03_network_forward.a_activation_function import (
    sigmoid,
    step,
    relu,
    identity_function,
    softmax,
)

ATOL = 1e-4


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.0]), np.array([0.5])),
        (np.array([1.0]), np.array([0.7310586])),
        (np.array([-1.0]), np.array([0.2689414])),
        (
            np.array([[0.0, 1.0], [-1.0, 2.0]]),
            np.array([[0.5, 0.7310586], [0.2689414, 0.8807971]]),
        ),
        (
            np.array([[[0.0], [1.0]], [[-1.0], [2.0]]]),
            np.array([[[0.5], [0.7310586]], [[0.2689414], [0.8807971]]]),
        ),
    ],
)
def test_sigmoid(
    x: np.typing.NDArray[np.floating], expected: np.typing.NDArray[np.floating]
) -> None:
    assert np.allclose(sigmoid(x), expected, atol=ATOL)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.0]), np.array([0], dtype=int)),
        (np.array([1.0]), np.array([1], dtype=int)),
        (np.array([-1.0]), np.array([0], dtype=int)),
        (np.array([[0.0, 1.0], [-1.0, 2.0]]), np.array([[0, 1], [0, 1]], dtype=int)),
        (
            np.array([[[0.0], [1.0]], [[-1.0], [2.0]]]),
            np.array([[[0], [1]], [[0], [1]]], dtype=int),
        ),
    ],
)
def test_step(
    x: np.typing.NDArray[np.floating], expected: np.typing.NDArray[np.floating]
) -> None:
    assert np.array_equal(step(x), expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.0]), np.array([0.0])),
        (np.array([1.0]), np.array([1.0])),
        (np.array([-1.0]), np.array([0.0])),
        (np.array([[0.0, 1.0], [-1.0, 2.0]]), np.array([[0.0, 1.0], [0.0, 2.0]])),
        (
            np.array([[[0.0], [1.0]], [[-1.0], [2.0]]]),
            np.array([[[0.0], [1.0]], [[0.0], [2.0]]]),
        ),
    ],
)
def test_relu(
    x: np.typing.NDArray[np.floating], expected: np.typing.NDArray[np.floating]
) -> None:
    assert np.allclose(relu(x), expected, atol=ATOL)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.0]), np.array([0.0])),
        (np.array([1.0]), np.array([1.0])),
        (np.array([-1.0]), np.array([-1.0])),
        (np.array([[0.0, 1.0], [-1.0, 2.0]]), np.array([[0.0, 1.0], [-1.0, 2.0]])),
        (
            np.array([[[0.0], [1.0]], [[-1.0], [2.0]]]),
            np.array([[[0.0], [1.0]], [[-1.0], [2.0]]]),
        ),
    ],
)
def test_identity_function(
    x: np.typing.NDArray[np.floating], expected: np.typing.NDArray[np.floating]
) -> None:
    assert np.allclose(identity_function(x), expected, atol=ATOL)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0.0]), np.array([1.0])),
        (np.array([1.0]), np.array([1.0])),
        (np.array([-1.0]), np.array([1.0])),
        (
            np.array([[0.0, 1.0], [-1.0, 2.0]]),
            np.array([[0.2689414, 0.7310586], [0.1192029, 0.8807971]]),
        ),
        (
            np.array([[[0.0], [1.0]], [[-1.0], [2.0]]]),
            np.array([[[1.0], [1.0]], [[1.0], [1.0]]]),
        ),
        (np.array([[1000.0, 1000.0]]), np.array([[0.5, 0.5]])),
    ],
)
def test_softmax(
    x: np.typing.NDArray[np.floating], expected: np.typing.NDArray[np.floating]
) -> None:
    assert np.allclose(softmax(x), expected, atol=ATOL)
