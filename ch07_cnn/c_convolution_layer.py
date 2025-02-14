import numpy as np
from numpy.typing import NDArray

from common.base import Layer


def conv_output_size(
    input_size: int, filter_size: int, stride: int = 1, pad: int = 0
) -> int:
    """Calculate the output size of the convolution operation.

    The formulation is:
    out_size = (input_size + 2 * pad - filter_size) // stride + 1

    Returns:
        int: Output size.
    """
    raise NotImplementedError


def im2col(
    input_data: NDArray[np.floating],
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
) -> NDArray[np.floating]:
    """Convert an image to a column for convolution.

    You have to avoid using 'for' loop, otherwise the processing time will be
    very long making train/test too slow.

    The intuitive principle of (1, c, h, w) imgage:
        get a filter-size img data over channel in order
                (i in h, j in w)
                        |-----------|
                        | |-----------|
                        | |  |------------|
                        |-|  |      w     |
                        |--|h  filter   |
                channel   |------------|

    How to calculate the dimention of columns:
        channel = C
        batch_size = N
        input_data = (N, C, H, W)
        filter_size = (C_out, C, FH, FW)

        H_out = (H - FH + 2 * pad) // stride + 1
        W_out = (W - FW + 2 * pad) // stride + 1

        col = (N * H_out * W_out, C * FH * FW)

    Parameters:
        input_data : NDArray[np.floating]
            4D array of input data. The shape is assumed to be
            (batch_size, channel, height, width).
        filter_h (int): Filter height.
        filter_w (int): Filter width.
        stride (int): Stride.
        pad (int): Padding.

    Returns:
        col : NDArray[np.floating]
            2D array, with above calculated dimention.
    """
    raise NotImplementedError


def col2im(
    col: NDArray[np.floating],
    input_shape: tuple[int, int, int, int],
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
    use_threading: bool = False,
) -> NDArray[np.floating]:
    """Convert a column to an image for convolution.

    You have to avoid using long 'for' loop, otherwise the processing time will
    be very long making train/test too slow.

    Parameters:
        col : NDArray[np.floating]
            2D arraym with shape (N * H_out * W_out, C * FH * FW), where:
                H_out = (H - FH + 2 * pad) // stride + 1
                W_out = (W - FW + 2 * pad) // stride + 1
            Usually, it is a gradient array in back propogation
        input_shape : tuple[int, int, int, int]
            Shape of the input data. The shape is assumed to be (N, C, H, W).
        filter_h (int): Filter height.
        filter_w (int): Filter width.
        stride (int): Stride.
        pad (int): Padding.
        use_threading (bool): Whether to use threading for parallel processing.

    Returns:
        NDArray[np.floating]: 4D array, with shape: (N, C, H, W).
    """
    raise NotImplementedError


class Conv2d(Layer):
    """Convolution layer.

    Usually, the input data is assumed to be a 4D array.  The 4D array is
    assumed to have the shape: (batch_size, channel, height, width).
    """

    def __init__(
        self,
        w: tuple[str, NDArray[np.floating]],
        b: tuple[str, NDArray[np.floating]],
        stride: int = 1,
        pad: int = 0,
    ) -> None:
        """Initialize the layer.

        Parameters:
            w: tuple[str, NDArray[np.floating]]
                Weights, with [name, width array]. The shape of array is:
                    (filter_num or output_channel, channel, filter_h, filter_w)
                    or (FN, C, FH, FW)
            b: tuple[str, NDArray[np.floating]]
                Biases, with [name, array]. The shape of array is: (1, filter_num)
            stride (int): Stride.
            pad (int): Padding.
        """
        self._w_name = w[0]
        self._b_name = b[0]
        self._params: dict[str, NDArray[np.floating]] = {w[0]: w[1], b[0]: b[1]}
        self._stride = stride
        self._pad = pad

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        raise NotImplementedError

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Dimention process:
            1). x -> im2col
                (N, C, H, W) -> (N * H_out * W_out, C * FH * FW)
                where:
                    H_out = (H - FH + 2 * pad) // stride + 1
                    W_out = (W - FW + 2 * pad) // stride + 1
            2). w (filter weight) -> w_col_T
                (FN, C, FH, FW) -> (C * FH * FW, FN)
            3). im2col @ w_col_T + b -> result -> out
                (N * H_out * W_out, C * FH * FW) @ (C * FH * FW, FN) + (1, Fn)
                -> (N * H_out * W_out, Fn) -> (N, F_n, H_out, W_out)

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D (N, C, H, W) array,
                where (batch_size, channel, height, width).

        Returns:
            convolution data : NDArray[np.floating]
                convolution result, with shape (N, FN, H_out, W_out).
        """
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Backward Propagation Dimension Process for CNN:

        -forward pass:
            1). x -> im2col
                (N, C, H, W) -> (N * H_out * W_out, C * FH * FW)
            2). w (filter weight) -> w_col_T
                (FN, C, FH, FW) -> (C * FH * FW, FN)
            3). im2col @ w_col_T + b -> result -> out
                (N * H_out * W_out, C * FH * FW) @ (C * FH * FW, FN) + (1, Fn)
                -> (N * H_out * W_out, Fn) -> (N, F_n, H_out, W_out)

        -backward pass:
            1) dout -> d_result
                (N, FN, H_out, W_out) -> (N * H_out * W_out, FN)
            2) d_b = sum(d_output) over row
                (1, FN) = sum((N * H_out * W_out, FN)) over row
            3) dw
                a. d_w_col_T = im2col.T @ d_result
                    (C * FH * FW, FN)
                    = (C * FH * FW, N * H_out * W_out) @ (N * H_out * W_out, FN)
                b. d_w_col_T -> d_w
                    -> (FN, C, FH, FW)
            4) dx
                a. d_im2col = d_result @ w_col_T.T
                    (N * H_out * W_out, C * FH * FW)
                    = (N * H_out * W_out, FN) @ (C * FH * FW, FN).T
                b. d_im2col ---col2im---> dx
                    -> (N, C, H, W)

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer, with the shape (N, Fn, H_out, W_out).

        Returns:
            NDArray[np.floating]:
                Gradient of the loss function with respect to the input of the
                layer. The shape is assumed to be a (N, C, H, W) array.
        """
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        raise NotImplementedError
