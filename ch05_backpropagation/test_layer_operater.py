import numpy as np
import pytest
from numpy.typing import NDArray

from ch05_backpropagation.a_layer_operator import (
    AddLayer,
    DivisorLayer,
    ExpoLayer,
    MulLayer,
)
from common.default_type_array import np_array


@pytest.mark.parametrize(
    "x, y, expected_forward, dout, expected_backward",
    [
        # 1D examples
        (
            np_array([1.0, 2.0]),
            np_array([3.0, 4.0]),
            np_array([4.0, 6.0]),
            np_array([1.0, 1.0]),
            (np_array([1.0, 1.0]), np_array([1.0, 1.0])),
        ),
        (
            np_array([0.0, 0.0]),
            np_array([0.0, 0.0]),
            np_array([0.0, 0.0]),
            np_array([0.0, 0.0]),
            (np_array([0.0, 0.0]), np_array([0.0, 0.0])),
        ),
        # 2D examples
        (
            np_array([[1.0, 2.0], [3.0, 4.0]]),
            np_array([[5.0, 6.0], [7.0, 8.0]]),
            np_array([[6.0, 8.0], [10.0, 12.0]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            (
                np_array([[1.0, 1.0], [1.0, 1.0]]),
                np_array([[1.0, 1.0], [1.0, 1.0]]),
            ),
        ),
        # 3D examples
        (
            np_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            np_array(
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
            ),
            np_array(
                [[[10.0, 12.0], [14.0, 16.0]], [[18.0, 20.0], [22.0, 24.0]]]
            ),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            (
                np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
                np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            ),
        ),
    ],
)
def test_add_layer(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: tuple[NDArray[np.floating], NDArray[np.floating]],
) -> None:
    layer = AddLayer()

    # Test forward pass
    out = layer.forward(x, y)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx, dy = layer.backward(dout)
    assert np.allclose(dx, expected_backward[0])
    assert np.allclose(dy, expected_backward[1])


@pytest.mark.parametrize(
    "x, y, expected_forward, dout, expected_backward",
    [
        # 1D examples
        (
            np_array([1.0, 2.0]),
            np_array([3.0, 4.0]),
            np_array([3.0, 8.0]),
            np_array([1.0, 1.0]),
            (np_array([3.0, 4.0]), np_array([1.0, 2.0])),
        ),
        (
            np_array([0.0, 0.0]),
            np_array([0.0, 0.0]),
            np_array([0.0, 0.0]),
            np_array([0.0, 0.0]),
            (np_array([0.0, 0.0]), np_array([0.0, 0.0])),
        ),
        # 2D examples
        (
            np_array([[1.0, 2.0], [3.0, 4.0]]),
            np_array([[5.0, 6.0], [7.0, 8.0]]),
            np_array([[5.0, 12.0], [21.0, 32.0]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            (
                np_array([[5.0, 6.0], [7.0, 8.0]]).T,
                np_array([[1.0, 2.0], [3.0, 4.0]]).T,
            ),
        ),
        # 3D examples
        (
            np_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            np_array(
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
            ),
            np_array(
                [[[9.0, 20.0], [33.0, 48.0]], [[65.0, 84.0], [105.0, 128.0]]]
            ),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            (
                np_array(
                    [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
                ).T,
                np_array(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
                ).T,
            ),
        ),
    ],
)
def test_mul_layer(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: tuple[NDArray[np.floating], NDArray[np.floating]],
) -> None:
    layer = MulLayer()

    # Test forward pass
    out = layer.forward(x, y)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx, dy = layer.backward(dout)
    assert np.allclose(dx, expected_backward[0])
    assert np.allclose(dy, expected_backward[1])


@pytest.mark.parametrize(
    "x, y, expected_forward, dout, expected_backward",
    [
        # 1D examples
        (
            np_array([1.0, 2.0]),
            np_array([3.0, 4.0]),
            np_array([1.0 / 3.0, 2.0 / 4.0]),
            np_array([1.0, 1.0]),
            (
                np_array([1.0 / 3.0, 1.0 / 4.0]),
                np_array([-1.0 / 9.0, -2.0 / 16.0]),
            ),
        ),
        (
            np_array([0.0, 0.0]),
            np_array([1.0, 1.0]),
            np_array([0.0, 0.0]),
            np_array([1.0, 1.0]),
            (np_array([1.0, 1.0]), np_array([0.0, 0.0])),
        ),
        # 2D examples
        (
            np_array([[1.0, 2.0], [3.0, 4.0]]),
            np_array([[5.0, 6.0], [7.0, 8.0]]),
            np_array([[1.0 / 5.0, 2.0 / 6.0], [3.0 / 7.0, 4.0 / 8.0]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            (
                np_array([[1.0 / 5.0, 1.0 / 6.0], [1.0 / 7.0, 1.0 / 8.0]]),
                np_array(
                    [[-1.0 / 25.0, -2.0 / 36.0], [-3.0 / 49.0, -4.0 / 64.0]]
                ),
            ),
        ),
        # 3D examples
        (
            np_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            np_array(
                [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
            ),
            np_array(
                [
                    [[1.0 / 9.0, 2.0 / 10.0], [3.0 / 11.0, 4.0 / 12.0]],
                    [[5.0 / 13.0, 6.0 / 14.0], [7.0 / 15.0, 8.0 / 16.0]],
                ]
            ),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            (
                np_array(
                    [
                        [[1.0 / 9.0, 1.0 / 10.0], [1.0 / 11.0, 1.0 / 12.0]],
                        [[1.0 / 13.0, 1.0 / 14.0], [1.0 / 15.0, 1.0 / 16.0]],
                    ]
                ),
                np_array(
                    [
                        [
                            [-1.0 / 81.0, -2.0 / 100.0],
                            [-3.0 / 121.0, -4.0 / 144.0],
                        ],
                        [
                            [-5.0 / 169.0, -6.0 / 196.0],
                            [-7.0 / 225.0, -8.0 / 256.0],
                        ],
                    ]
                ),
            ),
        ),
    ],
)
def test_divisor_layer(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: tuple[NDArray[np.floating], NDArray[np.floating]],
) -> None:
    layer = DivisorLayer()

    # Test forward pass
    out = layer.forward(x, y)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx, dy = layer.backward(dout)
    assert np.allclose(dx, expected_backward[0])
    assert np.allclose(dy, expected_backward[1])


@pytest.mark.parametrize(
    "x, expected_forward, dout, expected_backward",
    [
        # 1D examples
        (
            np_array([0.1, 0.2]),
            np_array([np.exp(0.1), np.exp(0.2)]),
            np_array([1.0, 1.0]),
            np_array([np.exp(0.1), np.exp(0.2)]),
        ),
        (
            np_array([0.0, 0.0]),
            np_array([np.exp(0.0), np.exp(0.0)]),
            np_array([1.0, 1.0]),
            np_array([np.exp(0.0), np.exp(0.0)]),
        ),
        # 2D examples
        (
            np_array([[0.1, 0.2], [0.3, 0.4]]),
            np_array([[np.exp(0.1), np.exp(0.2)], [np.exp(0.3), np.exp(0.4)]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            np_array([[np.exp(0.1), np.exp(0.2)], [np.exp(0.3), np.exp(0.4)]]),
        ),
        # 3D examples
        (
            np_array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            np_array(
                [
                    [[np.exp(0.1), np.exp(0.2)], [np.exp(0.3), np.exp(0.4)]],
                    [[np.exp(0.5), np.exp(0.6)], [np.exp(0.7), np.exp(0.8)]],
                ]
            ),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            np_array(
                [
                    [[np.exp(0.1), np.exp(0.2)], [np.exp(0.3), np.exp(0.4)]],
                    [[np.exp(0.5), np.exp(0.6)], [np.exp(0.7), np.exp(0.8)]],
                ]
            ),
        ),
    ],
)
def test_expo_layer(
    x: NDArray[np.floating],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: NDArray[np.floating],
) -> None:
    layer = ExpoLayer()

    # Test forward pass
    out = layer.forward(x)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)
