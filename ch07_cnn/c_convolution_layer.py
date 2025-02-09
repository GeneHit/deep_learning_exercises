import numpy as np
from numpy.typing import NDArray

from common.base import Layer
from common.default_type_array import np_zeros


def conv_output_size(
    input_size: int, filter_size: int, stride: int = 1, pad: int = 0
) -> int:
    """Calculate the output size of the convolution operation.

    The formulation is:
    out_size = (input_size + 2 * pad - filter_size) // stride + 1

    Returns:
        int: Output size.
    """
    return (input_size + 2 * pad - filter_size) // stride + 1


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
    n, c, h, w = input_data.shape
    h_out = conv_output_size(
        input_size=h, filter_size=filter_h, stride=stride, pad=pad
    )
    w_out = conv_output_size(
        input_size=w, filter_size=filter_w, stride=stride, pad=pad
    )

    # pad the height and weigth -> (N, C, Paded_H, Paded_W)
    padded_img = np.pad(
        array=input_data, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)]
    )

    # Use as_strided to create sliding windows for the convolution
    img_strides = padded_img.strides
    col = np.lib.stride_tricks.as_strided(
        padded_img,
        shape=(n, h_out, w_out, c, filter_h, filter_w),
        strides=(
            img_strides[0],
            stride * img_strides[2],
            stride * img_strides[3],
            img_strides[1],
            img_strides[2],
            img_strides[3],
        ),
    )

    # reshape the col to a new 2-D array
    return col.reshape(n * h_out * w_out, c * filter_h * filter_w)


def col2im(
    col: NDArray[np.floating],
    input_shape: tuple[int, int, int, int],
    filter_h: int,
    filter_w: int,
    stride: int,
    pad: int,
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

    Returns:
        NDArray[np.floating]: 4D array, with shape: (N, C, H, W).
    """
    n, c, h, w = input_shape
    h_out = conv_output_size(
        input_size=h, filter_size=filter_h, stride=stride, pad=pad
    )
    w_out = conv_output_size(
        input_size=w, filter_size=filter_w, stride=stride, pad=pad
    )

    assert col.shape == (n * h_out * w_out, c * filter_h * filter_w)
    # (N * H_out * W_out, C * FH * FW)
    # -> (N, H_out, W_out, C, FH, FW) -> (N, C, FH, FW, H_out, W_out)
    shapped_col = col.reshape(n, h_out, w_out, c, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    # increased by `stride - 1` to have enough space.
    padded_img = np_zeros(
        shape=(n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1)
    )
    # use filter_h/filter_w instead out_h/out_w to speed up.
    for y in range(filter_h):
        y_max = y + stride * h_out
        for x in range(filter_w):
            x_max = x + stride * w_out
            padded_img[:, :, y:y_max:stride, x:x_max:stride] += shapped_col[
                :, :, y, x, :, :
            ]

    # Return the image with padding removed (to match the original dimensions)
    return padded_img[:, :, pad : (pad + h), pad : (pad + w)]


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

        self._dw: NDArray[np.floating] | None = None
        self._db: NDArray[np.floating] | None = None

        # store the forward value for backward
        self._img_col: NDArray[np.floating] | None = None
        self._w_col_t: NDArray[np.floating] | None = None
        self._x_h: int | None = None
        self._x_w: int | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return self._params

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
        weight = self._params[self._w_name]
        f_n, c, f_h, f_w = weight.shape
        n, c, self._x_h, self._x_w = x.shape
        h_out, w_out = self._get_conv_height_and_width()

        # im2col(N*H_out*W_out, C*FH*FW)
        self._img_col = im2col(
            input_data=x,
            filter_h=f_h,
            filter_w=f_w,
            stride=self._stride,
            pad=self._pad,
        )
        self._w_col_t = weight.reshape(f_n, c * f_h * f_w).T
        # (N * H_out * W_out, FN)
        result: NDArray[np.floating] = (
            np.dot(self._img_col, self._w_col_t) + self._params[self._b_name]
        )
        # -> (N, H_out, W_out, FN) -> (N, FN, H_out, W_out)
        return result.reshape(n, h_out, w_out, f_n).transpose(0, 3, 1, 2)

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
        assert self._img_col is not None
        assert self._w_col_t is not None
        assert self._x_h is not None and self._x_w is not None
        n, f_n, h_out, w_out = dout.shape
        f_n, c, f_h, f_w = self._params[self._w_name].shape

        # 1. dout -> d_result
        d_result = dout.transpose(0, 2, 3, 1).reshape(n * h_out * w_out, f_n)
        # 2. dw (c * f_h * f_w, f_n)
        d_w_col_t: NDArray[np.floating] = np.dot(self._img_col.T, d_result)
        self._dw = d_w_col_t.T.reshape(f_n, c, f_h, f_w)
        del d_w_col_t
        # 3. db (1, f_n)
        self._db = np.sum(d_result, axis=0)
        # 4. d_x (n, c, h, w)
        d_img_col = np.dot(d_result, self._w_col_t.T)
        # free memory
        self._img_col = None
        self._w_col_t = None
        del d_result
        return col2im(
            col=d_img_col,
            input_shape=(n, c, self._x_h, self._x_w),
            filter_h=f_h,
            filter_w=f_w,
            stride=self._stride,
            pad=self._pad,
        )

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        assert self._dw is not None and self._db is not None
        return {self._w_name: self._dw, self._b_name: self._db}

    def _get_conv_height_and_width(self) -> tuple[int, int]:
        assert self._x_h is not None and self._x_w is not None
        _, _, f_h, f_w = self._params[self._w_name].shape
        h_out = conv_output_size(
            input_size=self._x_h,
            filter_size=f_h,
            stride=self._stride,
            pad=self._pad,
        )
        w_out = conv_output_size(
            input_size=self._x_w,
            filter_size=f_w,
            stride=self._stride,
            pad=self._pad,
        )
        return h_out, w_out
