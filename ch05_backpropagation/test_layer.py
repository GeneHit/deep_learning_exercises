import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import softmax

from ch04_network_learning.c_numerical_gradient import numerical_gradient
from ch05_backpropagation.b_layer import (
    Affine,
    ReLU,
    Sigmoid,
    Softmax,
    SoftmaxWithLoss,
)
from common.default_type_array import get_default_type, np_array, np_randn
from common.utils import assert_layer_parameter_type

ATOL = 1e-4


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
    assert out.dtype == get_default_type()
    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)
    assert dx.dtype == get_default_type()


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
    assert out.dtype == get_default_type()

    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)
    assert dx.dtype == get_default_type()


@pytest.mark.parametrize(
    "x, W, b, expected_forward, dout, expected_backward, expected_dW, expected_db",
    [
        # 2D examples
        (
            np_array([[1.0, 2.0]]),
            ("W", np_array([[1.0, 2.0], [3.0, 4.0]])),
            ("b", np_array([1.0, 2.0])),
            np_array([[8.0, 12.0]]),
            np_array([[1.0, 1.0]]),
            np_array([[3.0, 7.0]]),
            np_array([[1.0, 1.0], [2.0, 2.0]]),
            np_array([[1.0, 1.0]]),
        ),
        (
            np_array([[1.0, 2.0], [3.0, 4.0]]),
            ("W", np_array([[1.0, 2.0], [3.0, 4.0]])),
            ("b", np_array([1.0, 2.0])),
            np_array([[8.0, 12.0], [16.0, 24.0]]),
            np_array([[1.0, 1.0], [1.0, 1.0]]),
            np_array([[3.0, 7.0], [3.0, 7.0]]),
            np_array([[4.0, 4.0], [6.0, 6.0]]),
            np_array([2.0, 2.0]),
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
    assert out.dtype == get_default_type()

    # Test backward pass
    dx = layer.backward(dout)
    assert np.allclose(dx, expected_backward)
    assert dx.dtype == get_default_type()
    param_grads = layer.param_grads()
    assert np.allclose(param_grads[W[0]], expected_dW)
    assert np.allclose(param_grads[b[0]], expected_db)
    assert_layer_parameter_type(layer)


@pytest.mark.parametrize(
    "x",
    [
        # 2D examples
        np_array([[1.0, 2.0, 3.0]]),
        np_array([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_softmax_layer(x: NDArray[np.floating]) -> None:
    layer = Softmax()

    # Test forward pass
    out = layer.forward(x)
    assert np.allclose(out, softmax(x, axis=-1), atol=ATOL)
    assert out.dtype == get_default_type()

    # Test backward pass
    dout = np_randn(out.shape)
    dx = layer.backward(dout)

    # get numerical gradient
    def f(x_f: NDArray[np.floating]) -> float:
        return float(np.sum(layer.forward(x_f) * dout))

    numerical_dx = numerical_gradient(f, x)

    # compare the numerical gradient and the backward gradient
    assert dx.dtype == get_default_type()
    assert np.allclose(dx, numerical_dx, atol=1e-2)


@pytest.mark.parametrize(
    "x, t, expected_loss, expected_backward",
    [
        # 2D examples
        (
            np_array([[0.3, 0.2, 0.5]]),
            np_array([[0, 0, 1]]),
            0.93983,
            np_array([0.31987, 0.28943, -0.60931]),
        ),
        (
            np_array([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]]),
            np_array([[0, 0, 1], [0, 1, 0]]),
            1.62956,
            np_array(
                [
                    [0.15993, 0.14471, -0.30465],
                    [0.12457, -0.24914, 0.12457],
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
    assert np.allclose(loss, expected_loss, atol=ATOL)

    # Test backward pass
    dx = layer.backward()
    assert dx.dtype == get_default_type()
    assert np.allclose(dx, expected_backward, atol=ATOL)
