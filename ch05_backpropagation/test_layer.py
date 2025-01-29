import numpy as np
import pytest
from numpy.typing import NDArray

from ch05_backpropagation.b_layer import Affine, ReLU, Sigmoid, SoftmaxWithLoss
from common.default_type_array import np_array, np_float


@pytest.mark.parametrize(
    "x, expected_forward, dout, expected_backward",
    [
        # 1D examples
        (
            np_array([1.0, -2.0, 3.0, -4.0]),
            np_array([1.0, 0.0, 3.0, 0.0]),
            np_array([1.0, 1.0, 1.0, 1.0]),
            np_array([1.0, 0.0, 1.0, 0.0]),
        ),
        (
            np_array([0.0, -1.0, 2.0, -3.0]),
            np_array([0.0, 0.0, 2.0, 0.0]),
            np_array([1.0, 1.0, 1.0, 1.0]),
            np_array([0.0, 0.0, 1.0, 0.0]),
        ),
        # 2D examples
        (
            np_array([[1.0, -2.0], [3.0, -4.0]]),
            np_array([[1.0, 0.0], [3.0, 0.0]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            np_array([[1.0, 0.0], [1.0, 0.0]]),
        ),
        # 3D examples
        (
            np_array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]]),
            np_array([[[1.0, 0.0], [3.0, 0.0]], [[5.0, 0.0], [7.0, 0.0]]]),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            np_array([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]),
        ),
    ],
)
def test_relu_layer(
    x: NDArray[np.floating],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: NDArray[np.floating],
) -> None:
    layer = ReLU()

    # Test forward pass
    out = layer.forward(x)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)


@pytest.mark.parametrize(
    "x, expected_forward, dout, expected_backward",
    [
        # 1D examples
        (
            np_array([0.0, 1.0, -1.0]),
            np_array(
                [
                    1 / (1 + np.exp(0.0)),
                    1 / (1 + np.exp(-1.0)),
                    1 / (1 + np.exp(1.0)),
                ]
            ),
            np_array([1.0, 1.0, 1.0]),
            np_array(
                [
                    1 / (1 + np.exp(0.0)) * (1 - 1 / (1 + np.exp(0.0))),
                    1 / (1 + np.exp(-1.0)) * (1 - 1 / (1 + np.exp(-1.0))),
                    1 / (1 + np.exp(1.0)) * (1 - 1 / (1 + np.exp(1.0))),
                ]
            ),
        ),
        (
            np_array([0.5, -0.5]),
            np_array([1 / (1 + np.exp(-0.5)), 1 / (1 + np.exp(0.5))]),
            np_array([1.0, 1.0]),
            np_array(
                [
                    1 / (1 + np.exp(-0.5)) * (1 - 1 / (1 + np.exp(-0.5))),
                    1 / (1 + np.exp(0.5)) * (1 - 1 / (1 + np.exp(0.5))),
                ]
            ),
        ),
        # 2D examples
        (
            np_array([[0.0, 1.0], [-1.0, 0.5]]),
            np_array(
                [
                    [1 / (1 + np.exp(0.0)), 1 / (1 + np.exp(-1.0))],
                    [1 / (1 + np.exp(1.0)), 1 / (1 + np.exp(-0.5))],
                ]
            ),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            np_array(
                [
                    [
                        1 / (1 + np.exp(0.0)) * (1 - 1 / (1 + np.exp(0.0))),
                        1 / (1 + np.exp(-1.0)) * (1 - 1 / (1 + np.exp(-1.0))),
                    ],
                    [
                        1 / (1 + np.exp(1.0)) * (1 - 1 / (1 + np.exp(1.0))),
                        1 / (1 + np.exp(-0.5)) * (1 - 1 / (1 + np.exp(-0.5))),
                    ],
                ]
            ),
        ),
        # 3D examples
        (
            np_array([[[0.0, 1.0], [-1.0, 0.5]], [[0.0, 1.0], [-1.0, 0.5]]]),
            np_array(
                [
                    [
                        [1 / (1 + np.exp(0.0)), 1 / (1 + np.exp(-1.0))],
                        [1 / (1 + np.exp(1.0)), 1 / (1 + np.exp(-0.5))],
                    ],
                    [
                        [1 / (1 + np.exp(0.0)), 1 / (1 + np.exp(-1.0))],
                        [1 / (1 + np.exp(1.0)), 1 / (1 + np.exp(-0.5))],
                    ],
                ]
            ),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            np_array(
                [
                    [
                        [
                            1 / (1 + np.exp(0.0)) * (1 - 1 / (1 + np.exp(0.0))),
                            1
                            / (1 + np.exp(-1.0))
                            * (1 - 1 / (1 + np.exp(-1.0))),
                        ],
                        [
                            1 / (1 + np.exp(1.0)) * (1 - 1 / (1 + np.exp(1.0))),
                            1
                            / (1 + np.exp(-0.5))
                            * (1 - 1 / (1 + np.exp(-0.5))),
                        ],
                    ],
                    [
                        [
                            1 / (1 + np.exp(0.0)) * (1 - 1 / (1 + np.exp(0.0))),
                            1
                            / (1 + np.exp(-1.0))
                            * (1 - 1 / (1 + np.exp(-1.0))),
                        ],
                        [
                            1 / (1 + np.exp(1.0)) * (1 - 1 / (1 + np.exp(1.0))),
                            1
                            / (1 + np.exp(-0.5))
                            * (1 - 1 / (1 + np.exp(-0.5))),
                        ],
                    ],
                ]
            ),
        ),
    ],
)
def test_sigmoid_layer(
    x: NDArray[np.floating],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: NDArray[np.floating],
) -> None:
    layer = Sigmoid()

    # Test forward pass
    out = layer.forward(x)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)


@pytest.mark.parametrize(
    "x, W, b, expected_forward, dout, expected_backward, expected_dW, expected_db",
    [
        # 1D examples
        (
            np_array([[1.0, 2.0]]),
            ("W", np_array([[1.0, 2.0], [3.0, 4.0]])),
            ("b", np_array([1.0, 2.0])),
            np_array([[8.0, 12.0]]),
            np_array([[1.0, 1.0]]),
            np_array([[4.0, 6.0]]),
            np_array([[1.0, 1.0], [2.0, 2.0]]),
            np_array([1.0, 1.0]),
        ),
        # 2D examples
        (
            np_array([[1.0, 2.0], [3.0, 4.0]]),
            ("W", np_array([[1.0, 2.0], [3.0, 4.0]])),
            ("b", np_array([1.0, 2.0])),
            np_array([[8.0, 12.0], [16.0, 22.0]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            np_array([[4.0, 6.0], [4.0, 6.0]]),
            np_array([[4.0, 4.0], [6.0, 6.0]]),
            np_array([2.0, 2.0]),
        ),
        # 3D examples
        (
            np_array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            ("W", np_array([[1.0, 2.0], [3.0, 4.0]])),
            ("b", np_array([1.0, 2.0])),
            np_array(
                [[[8.0, 12.0], [16.0, 22.0]], [[24.0, 32.0], [32.0, 42.0]]]
            ),
            np_array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            np_array([[[4.0, 6.0], [4.0, 6.0]], [[4.0, 6.0], [4.0, 6.0]]]),
            np_array([[16.0, 16.0], [24.0, 24.0]]),
            np_array([4.0, 4.0]),
        ),
    ],
)
def test_affine_layer(
    x: NDArray[np.floating],
    W: tuple[str, NDArray[np.floating]],
    b: tuple[str, NDArray[np.floating]],
    expected_forward: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_backward: NDArray[np.floating],
    expected_dW: NDArray[np.floating],
    expected_db: NDArray[np.floating],
) -> None:
    layer = Affine(W, b)

    # Test forward pass
    out = layer.forward(x)
    assert np.allclose(out, expected_forward)

    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)
    param_grads = layer.param_grads()
    assert np.allclose(param_grads[W[0]], expected_dW)
    assert np.allclose(param_grads[b[0]], expected_db)


@pytest.mark.parametrize(
    "x, t, expected_loss, expected_backward",
    [
        # 1D examples
        (
            np_array([0.3, 0.2, 0.5]),
            np_array([0, 0, 1]),
            -np.log(np_float(0.5)),
            np_array([0.1, 0.06666667, -0.16666667]),
        ),
        (
            np_array([0.1, 0.8, 0.1]),
            np_array([0, 1, 0]),
            -np.log(np_float(0.8)),
            np_array([0.03333333, -0.06666667, 0.03333333]),
        ),
        # 2D examples
        (
            np_array([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]]),
            np_array([[0, 0, 1], [0, 1, 0]]),
            -np.log(np_float(0.5)) / 2 - np.log(np_float(0.8)) / 2,
            np_array(
                [
                    [0.1, 0.06666667, -0.16666667],
                    [0.03333333, -0.06666667, 0.03333333],
                ]
            ),
        ),
        # 3D examples
        (
            np_array(
                [
                    [[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]],
                    [[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]],
                ]
            ),
            np_array([[[0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 1, 0]]]),
            (-np.log(np_float(0.5)) / 2 - np.log((0.8)) / 2) * 2,
            np_array(
                [
                    [
                        [0.1, 0.06666667, -0.16666667],
                        [0.03333333, -0.06666667, 0.03333333],
                    ],
                    [
                        [0.1, 0.06666667, -0.16666667],
                        [0.03333333, -0.06666667, 0.03333333],
                    ],
                ]
            ),
        ),
    ],
)
def test_softmax_with_loss_layer(
    x: NDArray[np.floating],
    t: NDArray[np.floating],
    expected_loss: float,
    expected_backward: NDArray[np.floating],
) -> None:
    layer = SoftmaxWithLoss()

    # Test forward pass
    loss = layer.forward_to_loss(x, t)
    assert np.allclose(loss, expected_loss)

    # Test backward pass
    dx = layer.backward()
    assert np.allclose(dx, expected_backward)
