import numpy as np
from numpy.typing import NDArray

from ch07_cnn.c_convolution_layer import col2im, conv_output_size, im2col
from common.base import Layer
from common.default_type_array import np_float, np_zeros


class MaxPool2d(Layer):
    """Max 2D pooling layer.

    Pooling operations, whether 2D or 3D, are applied independently to
    each channel of the input data. This means that the pooling operation
    processes the spatial dimensions (height and width for 2D pooling,
    depth, height, and width for 3D pooling) but does not mix or combine
    information across different channels.
    """

    def __init__(
        self, kenel_size: tuple[int, int], stride: int = 2, pad: int = 0
    ) -> None:
        """Initialize the MaxPool2d layer.

        Parameters:
            kenel_size: tuple[int, int]
                The size of the pooling window. The tuple should have two
                integers: (height, width).
            stride: int
                The stride of the pooling operation.
            pad: int
                The padding applied to the input data.
        """
        self._kenel_size = kenel_size
        self._stride = stride
        self._pad = pad

        # the data in forward for backward computation
        self._x_h: int | None = None
        self._x_w: int | None = None
        self._argmax_col: NDArray[np.floating] | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Tips: can use the im2col to have a col and then apply the max operation.

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        n, c, self._x_h, self._x_w = x.shape
        h_out, w_out = self._get_output_height_and_width()
        f_h, f_w = self._kenel_size

        # (N, C, H, W)
        # -> (N * H_out * W_out, C*FH*FW) -> (N * H_out * W_out * C, FH*FW)
        col = im2col(
            input_data=x,
            filter_h=f_h,
            filter_w=f_w,
            stride=self._stride,
            pad=self._pad,
        ).reshape(-1, f_h * f_w)
        # (N * H_out * W_out * C, 1)
        max_col: NDArray[np.floating] = np.max(col, axis=1)
        self._argmax_col = np.argmax(col, axis=1)
        # -> (N, H_out, W_out, C) -> (N, C, H_out, W_out)
        return max_col.reshape(n, h_out, w_out, c).transpose(0, 3, 1, 2)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        assert self._argmax_col is not None
        assert self._x_h is not None
        assert self._x_w is not None
        n, c, h_out, w_out = dout.shape
        f_h, f_w = self._kenel_size
        multi_n_hout_wout = n * h_out * w_out
        multi_fh_fw = f_h * f_w

        # -> (N, H_out, W_out, C) -> (N * H_out * W_out * C, 1)
        d_max_col = dout.transpose(0, 2, 3, 1).reshape(multi_n_hout_wout * c, 1)

        # (N * H_out * W_out * C, FH * FW)
        d_col = np_zeros(shape=(multi_n_hout_wout * c, multi_fh_fw))
        # the argmax_col (N * H_out * W_out * C, 1) has store the indice of max
        # the other indices' gradient is 0.
        d_col[np.arange(self._argmax_col.shape[0]), self._argmax_col] = (
            d_max_col.flatten()
        )

        # (N * H_out * W_out, C * FH * FW)
        d_img_col = d_col.reshape(multi_n_hout_wout, c * multi_fh_fw)

        # free memory
        del d_max_col
        del d_col

        # (N, C, x_H, x_W)
        dx = col2im(
            col=d_img_col,
            input_shape=(n, c, self._x_h, self._x_w),
            filter_h=f_h,
            filter_w=f_w,
            stride=self._stride,
            pad=self._pad,
        )

        self._argmax_col = None
        return dx

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        # There are no parameters to update in the MaxPool2d layer
        return {}

    def _get_output_height_and_width(self) -> tuple[int, int]:
        assert self._x_h is not None and self._x_w is not None
        h_out = conv_output_size(
            input_size=self._x_h,
            filter_size=self._kenel_size[0],
            stride=self._stride,
            pad=self._pad,
        )
        w_out = conv_output_size(
            input_size=self._x_w,
            filter_size=self._kenel_size[1],
            stride=self._stride,
            pad=self._pad,
        )
        return h_out, w_out


class AvgPool2d(Layer):
    """Average pooling layer.

    Pooling operations, whether 2D or 3D, are applied independently to
    each channel of the input data. This means that the pooling operation
    processes the spatial dimensions (height and width for 2D pooling,
    depth, height, and width for 3D pooling) but does not mix or combine
    information across different channels.
    """

    def __init__(
        self, kenel_size: tuple[int, int], stride: int = 2, pad: int = 0
    ) -> None:
        self._kenel_size = kenel_size
        self._stride = stride
        self._pad = pad

        # the data in forward for backward computation
        self._x_h: int | None = None
        self._x_w: int | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Tips: can use the im2col to have a col and then apply the avg operation.

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        n, c, self._x_h, self._x_w = x.shape
        h_out, w_out = self._get_output_height_and_width()
        f_h, f_w = self._kenel_size

        # (N, C, H, W)
        # -> (N * H_out * W_out, C*FH*FW) -> (N * H_out * W_out * C, FH*FW)
        col = im2col(
            input_data=x,
            filter_h=f_h,
            filter_w=f_w,
            stride=self._stride,
            pad=self._pad,
        ).reshape(-1, f_h * f_w)
        # (N * H_out * W_out * C, 1)
        avg_col: NDArray[np.floating] = np.average(col, axis=1)
        # -> (N, H_out, W_out, C) -> (N, C, H_out, W_out)
        return avg_col.reshape(n, h_out, w_out, c).transpose(0, 3, 1, 2)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        assert self._x_h is not None
        assert self._x_w is not None
        n, c, h_out, w_out = dout.shape
        f_h, f_w = self._kenel_size
        multi_n_hout_w_out = n * h_out * w_out
        multi_fh_fw = f_h * f_w

        # (N, C, H_out, W_out)
        # -> (N, H_out, W_out, C) -> (N * H_out * W_out * C, 1)
        d_avg_col = dout.transpose(0, 2, 3, 1).reshape(
            multi_n_hout_w_out * c, 1
        )

        # (N * H_out * W_out * C, FH * FW)
        d_col = np.tile(d_avg_col, (1, multi_fh_fw)) / np_float(multi_fh_fw)

        # (N * H_out * W_out, C * FH * FW)
        d_img_col = d_col.reshape(multi_n_hout_w_out, c * multi_fh_fw)

        # free memory
        del d_avg_col
        del d_col

        # (n, c, x_h, x_w)
        dx = col2im(
            col=d_img_col,
            input_shape=(n, c, self._x_h, self._x_w),
            filter_h=f_h,
            filter_w=f_w,
            stride=self._stride,
            pad=self._pad,
        )

        return dx

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        # There are no parameters to update in the AvgPool2d layer
        return {}

    def _get_output_height_and_width(self) -> tuple[int, int]:
        assert self._x_h is not None and self._x_w is not None
        h_out = conv_output_size(
            input_size=self._x_h,
            filter_size=self._kenel_size[0],
            stride=self._stride,
            pad=self._pad,
        )
        w_out = conv_output_size(
            input_size=self._x_w,
            filter_size=self._kenel_size[1],
            stride=self._stride,
            pad=self._pad,
        )
        return h_out, w_out


class Flatten(Layer):
    """Flatten layer.

    This operation is crucial for transitioning from convolutional layers to
    fully connected layers. The Flatten layer reshapes the input data into a
    2D array, with shape (batch_size, n).
    How to calculate n:
        n = channel * height * width
    """

    def __init__(self) -> None:
        self._x_shape: tuple[int, ...] | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        return {}

    def train_flag(self, flag: bool) -> None:
        """See the base class."""
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).

        Returns:
            NDArray[np.floating]: The reshaped array, with shape (batch_size, n).
        """
        self._x_shape = x.shape
        # -1 means it will calulate the col automaticlly
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 2D array:
                    (batch_size, n).

        Returns:
            dout: NDArray[np.floating]
                Gradient of the loss function with respect to the output of
                the layer. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).
        """
        assert self._x_shape is not None
        result: NDArray[np.floating] = dout.reshape(*self._x_shape)
        return result

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        # There are no parameters to update in the Flatten layer
        return {}
