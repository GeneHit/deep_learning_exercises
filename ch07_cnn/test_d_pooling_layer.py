import numpy as np
import pytest
from numpy.typing import NDArray

from ch07_cnn.d_pooling_layer import AvgPool2d, Flatten, MaxPool2d


class TestMaxPool2d:
    """Tests for MaxPool2d layer."""

    @pytest.mark.parametrize(
        "kenel_size, stride, pad, input_x, expected_output",
        [
            # Example 1: Simple 2x2 pooling, no padding
            (
                (2, 2),
                2,
                0,
                np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),  # input_x
                np.array([[[[5, 6], [8, 9]]]]),  # expected_output
            ),
            # Example 2: Single channel, larger stride
            (
                (2, 2),
                3,
                0,
                np.array(
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
                np.array([[[[6, 8], [14, 16]]]]),
            ),
        ],
    )
    def test_forward(
        self,
        kenel_size: tuple[int, int],
        stride: int,
        pad: int,
        input_x: NDArray[np.floating],
        expected_output: NDArray[np.floating],
    ) -> None:
        """Test the forward function of MaxPool2d."""
        layer = MaxPool2d(kenel_size, stride, pad)
        output = layer.forward(input_x)
        np.testing.assert_array_almost_equal(output, expected_output)

    @pytest.mark.parametrize(
        "kenel_size, stride, pad, input_x, dout, expected_dx",
        [
            # Example 1: Backpropagation for 2x2 pooling, no padding
            (
                (2, 2),
                2,
                0,
                np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),  # input_x
                np.array([[[[1, 1], [1, 1]]]]),  # dout
                np.array([[[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]]),  # expected_dx
            ),
            # Example 2: Backpropagation for a larger input
            (
                (2, 2),
                2,
                0,
                np.array([[[[1, 3, 1], [4, 2, 6], [7, 5, 9]]]]),
                np.array([[[[2, 3], [4, 5]]]]),
                np.array([[[[0, 0, 0], [2, 0, 3], [4, 0, 5]]]]),
            ),
        ],
    )
    def test_backward(
        self,
        kenel_size: tuple[int, int],
        stride: int,
        pad: int,
        input_x: NDArray[np.floating],
        dout: NDArray[np.floating],
        expected_dx: NDArray[np.floating],
    ) -> None:
        """Test the backward function of MaxPool2d."""
        layer = MaxPool2d(kenel_size, stride, pad)
        # Run forward pass to store indices for backward
        layer.forward(input_x)
        dx = layer.backward(dout)
        np.testing.assert_array_almost_equal(dx, expected_dx)


class TestAvgPool2d:
    """Tests for AvgPool2d layer."""

    @pytest.mark.parametrize(
        "kenel_size, stride, pad, input_x, expected_output",
        [
            # Example 1: Simple 2x2 pooling, no padding
            (
                (2, 2),
                2,
                0,
                np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),  # input_x
                np.array([[[[3, 4.5], [7.5, 9]]]]),  # expected_output
            ),
            # Example 2: Single channel, stride = 3
            (
                (2, 2),
                3,
                0,
                np.array(
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
                np.array([[[[3.5, 5.5], [11.5, 13.5]]]]),
            ),
        ],
    )
    def test_forward(
        self,
        kenel_size: tuple[int, int],
        stride: int,
        pad: int,
        input_x: NDArray[np.floating],
        expected_output: NDArray[np.floating],
    ) -> None:
        """Test the forward function of AvgPool2d."""
        layer = AvgPool2d(kenel_size, stride, pad)
        output = layer.forward(input_x)
        np.testing.assert_array_almost_equal(output, expected_output)

    @pytest.mark.parametrize(
        "kenel_size, stride, pad, input_x, dout, expected_dx",
        [
            # Example 1: Backpropagation for 2x2 pooling, no padding
            (
                (2, 2),
                2,
                0,
                np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),  # input_x
                np.array([[[[1, 1], [1, 1]]]]),  # dout
                np.array(
                    [[[[0.25, 0.25, 0], [0.25, 0.25, 0], [0, 0, 0]]]]
                ),  # expected_dx
            ),
            # Example 2: Backpropagation for larger input
            (
                (2, 2),
                2,
                0,
                np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
                np.array([[[[2, 3], [4, 5]]]]),
                np.array([[[[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]]]]),
            ),
        ],
    )
    def test_backward(
        self,
        kenel_size: tuple[int, int],
        stride: int,
        pad: int,
        input_x: NDArray[np.floating],
        dout: NDArray[np.floating],
        expected_dx: NDArray[np.floating],
    ) -> None:
        """Test the backward function of AvgPool2d."""
        layer = AvgPool2d(kenel_size, stride, pad)
        # Run forward pass to store indices for backward
        layer.forward(input_x)
        dx = layer.backward(dout)
        np.testing.assert_array_almost_equal(dx, expected_dx)


class TestFlatten:
    """Tests for Flatten layer."""

    @pytest.mark.parametrize(
        "input_x, expected_output",
        [
            # Example 1: Single batch, single channel
            (
                np.array([[[[1, 2], [3, 4]]]]),  # input_x
                np.array([[1, 2, 3, 4]]),  # expected_output
            ),
            # Example 2: Multiple batches, multiple channels
            (
                np.array([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]),  # input_x
                np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),  # expected_output
            ),
        ],
    )
    def test_forward(
        self,
        input_x: NDArray[np.floating],
        expected_output: NDArray[np.floating],
    ) -> None:
        """Test the forward function of Flatten."""
        layer = Flatten()
        output = layer.forward(input_x)
        np.testing.assert_array_equal(output, expected_output)

    @pytest.mark.parametrize(
        "input_x, dout, expected_dx",
        [
            # Example 1: Single batch, single channel
            (
                np.array([[[[1, 2], [3, 4]]]]),  # input_x
                np.array([[1, 2, 3, 4]]),  # dout
                np.array([[[[1, 2], [3, 4]]]]),  # expected_dx
            ),
            # Example 2: Multiple batches, multiple channels
            (
                np.array([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]),  # input_x
                np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),  # dout
                np.array(
                    [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]
                ),  # expected_dx
            ),
        ],
    )
    def test_backward(
        self,
        input_x: NDArray[np.floating],
        dout: NDArray[np.floating],
        expected_dx: NDArray[np.floating],
    ) -> None:
        """Test the backward function of Flatten."""
        layer = Flatten()
        # Run forward pass to store input shape
        layer.forward(input_x)
        dx = layer.backward(dout)
        np.testing.assert_array_equal(dx, expected_dx)
