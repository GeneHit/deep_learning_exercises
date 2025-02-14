import numpy as np
import pytest
from numpy.typing import NDArray

from ch07_cnn.c_convolution_layer import (
    Conv2d,
    col2im,
    conv_output_size,
    im2col,
)
from common.default_type_array import np_array

ATOL = 1e-5


@pytest.mark.parametrize(
    "input_size, filter_size, stride, pad, expected_output_size",
    [
        (5, 3, 1, 0, 3),
        (5, 3, 1, 1, 5),
        (5, 3, 2, 0, 2),
        (5, 3, 2, 1, 3),
        (7, 3, 1, 0, 5),
        (7, 3, 1, 1, 7),
        (7, 3, 2, 0, 3),
        (7, 3, 2, 1, 4),
    ],
)
def test_conv_output_size(
    input_size: int,
    filter_size: int,
    stride: int,
    pad: int,
    expected_output_size: int,
) -> None:
    output_size = conv_output_size(input_size, filter_size, stride, pad)
    assert output_size == expected_output_size


@pytest.mark.parametrize(
    "input_data, filter_h, filter_w, stride, pad, expected_output_shape, expected_result",
    [
        (
            np_array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            2,
            2,
            1,
            0,
            (4, 4),
            np_array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]]),
        ),
        (
            np_array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            2,
            2,
            1,
            1,
            (16, 4),
            np_array(
                [
                    [0, 0, 0, 1],
                    [0, 0, 1, 2],
                    [0, 0, 2, 3],
                    [0, 0, 3, 0],
                    [0, 1, 0, 4],
                    [1, 2, 4, 5],
                    [2, 3, 5, 6],
                    [3, 0, 6, 0],
                    [0, 4, 0, 7],
                    [4, 5, 7, 8],
                    [5, 6, 8, 9],
                    [6, 0, 9, 0],
                    [0, 7, 0, 0],
                    [7, 8, 0, 0],
                    [8, 9, 0, 0],
                    [9, 0, 0, 0],
                ]
            ),
        ),
        (
            np_array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            2,
            2,
            2,
            0,
            (1, 4),
            np_array([[1, 2, 4, 5]]),
        ),
        (
            np_array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            2,
            2,
            2,
            1,
            (4, 4),
            np_array([[0, 0, 0, 1], [0, 0, 2, 3], [0, 4, 0, 7], [5, 6, 8, 9]]),
        ),
    ],
)
def test_im2col(
    input_data: NDArray[np.floating],
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
    expected_output_shape: tuple[int, int],
    expected_result: NDArray[np.floating],
) -> None:
    col = im2col(input_data, filter_h, filter_w, stride, pad)
    assert col.shape == expected_output_shape
    assert np.allclose(col, expected_result)


@pytest.mark.parametrize(
    "col, input_shape, filter_h, filter_w, stride, pad, expected_result",
    [
        (
            np_array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]]),
            (1, 1, 3, 3),
            2,
            2,
            1,
            0,
            np_array([[[[1, 4, 3], [8, 20, 12], [7, 16, 9]]]]),
        ),
        (
            np_array(
                [
                    [0, 0, 0, 1],
                    [0, 0, 1, 2],
                    [0, 0, 2, 3],
                    [0, 0, 3, 0],
                    [0, 1, 0, 4],
                    [1, 2, 4, 5],
                    [2, 3, 5, 6],
                    [3, 0, 6, 0],
                    [0, 4, 0, 7],
                    [4, 5, 7, 8],
                    [5, 6, 8, 9],
                    [6, 0, 9, 0],
                    [0, 7, 0, 0],
                    [7, 8, 0, 0],
                    [8, 9, 0, 0],
                    [9, 0, 0, 0],
                ]
            ),
            (1, 1, 3, 3),
            2,
            2,
            1,
            1,
            np_array([[[[4, 8, 12], [16, 20, 24], [28, 32, 36]]]]),
        ),
        (
            np_array([[1, 2, 4, 5]]),
            (1, 1, 3, 3),
            2,
            2,
            2,
            0,
            np_array([[[[1, 2, 0], [4, 5, 0], [0, 0, 0]]]]),
        ),
        (
            np_array([[0, 0, 0, 1], [0, 0, 2, 3], [0, 4, 0, 7], [4, 5, 7, 8]]),
            (1, 1, 3, 3),
            2,
            2,
            2,
            1,
            np_array([[[[1, 2, 3], [4, 4, 5], [7, 7, 8]]]]),
        ),
    ],
)
def test_col2im(
    col: NDArray[np.floating],
    input_shape: tuple[int, int, int, int],
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
    expected_result: NDArray[np.floating],
) -> None:
    img = col2im(col, input_shape, filter_h, filter_w, stride, pad)
    assert np.allclose(img, expected_result, atol=ATOL)


@pytest.mark.parametrize(
    "w0, b0, stride, pad, input_x, dout, expected_output, expected_dw, expected_db, expected_dx",
    [
        (
            ["W0", np_array([[[[1, 0], [0, -1]]]])],
            ["b0", np_array([0])],
            1,  # stride
            0,  # pad
            # input_x
            np_array(
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                        ]
                    ]
                ]
            ),
            # dout
            np_array([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]),
            # expected_output
            np_array([[[[-5, -5, -5], [-5, -5, -5], [-5, -5, -5]]]]),
            # expected_dw
            np_array([[[[54, 63], [90, 99]]]]),
            9,  # expected_db
            # expected_dx
            np_array(
                [[[1, 1, 1, 0], [1, 0, 0, -1], [1, 0, 0, -1], [0, -1, -1, -1]]]
            ),
        ),
        (
            ("w0", np_array([[[[1, 0], [0, -1]]], [[[-1, 1], [1, -1]]]])),
            ("b0", np_array([1, -1])),
            1,  # stride
            0,  # pad
            # input_x
            np_array(
                [
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[9, 8, 7], [6, 5, 4], [3, 2, 1]]],
                ]
            ),
            np.ones((2, 2, 2, 2)),  # dout
            # expected_output
            np_array(
                [
                    [[[-3, -3], [-3, -3]], [[-1, -1], [-1, -1]]],
                    [[[5, 5], [5, 5]], [[-1, -1], [-1, -1]]],
                ]
            ),
            # expected_dw
            np_array([[[[40, 40], [40, 40]]], [[[40, 40], [40, 40]]]]),
            # expected_db
            np_array([8, 8]),
            # expected_dx
            np_array(
                [
                    [[[0, 1, 1], [1, 0, -1], [1, -1, -2]]],
                    [[[0, 1, 1], [1, 0, -1], [1, -1, -2]]],
                ]
            ),
        ),
    ],
)
def test_convolution(
    w0: tuple[str, NDArray[np.floating]],
    b0: tuple[str, NDArray[np.floating]],
    stride: int,
    pad: int,
    input_x: NDArray[np.floating],
    dout: NDArray[np.floating],
    expected_output: NDArray[np.floating],
    expected_dw: NDArray[np.floating],
    expected_db: NDArray[np.floating],
    expected_dx: NDArray[np.floating],
) -> None:
    conv = Conv2d(w0, b0, stride, pad)

    output_y = conv.forward(input_x)
    dx = conv.backward(dout)
    grads = conv.param_grads()
    dw = grads[w0[0]]
    db = grads[b0[0]]

    assert np.allclose(output_y, expected_output, atol=ATOL)
    assert np.allclose(dw, expected_dw, atol=ATOL)
    assert np.allclose(db, expected_db, atol=ATOL)
    assert np.allclose(dx, expected_dx, atol=ATOL)
