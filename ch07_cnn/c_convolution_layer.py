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
) -> NDArray[np.floating]:
    """Convert a column to an image for convolution.

    Parameters:
        col : NDArray[np.floating]
            2D arraym with shape (N * H_out * W_out, C * FH * FW), where:
                H_out = (H - FH + 2 * pad) // stride + 1
                W_out = (W - FW + 2 * pad) // stride + 1
        input_shape : tuple[int, int, int, int]
            Shape of the input data. The shape is assumed to be (N, C, H, W).
        filter_h (int): Filter height.
        filter_w (int): Filter width.
        stride (int): Stride.
        pad (int): Padding.

    Returns:
        reNDArray[np.floating]: 4D array, with shape: (N, C, H, W).
    """
    raise NotImplementedError


class Conv2d(Layer):
    """Convolution layer.

    Usually, the input data is assumed to be a 4D array.  The 4D array is
    assumed to have the shape: (batch_size, channel, height, width).
    """

    def __init__(
        self,
        W: tuple[str, NDArray[np.floating]],
        b: tuple[str, NDArray[np.floating]],
        stride: int = 1,
        pad: int = 0,
    ) -> None:
        """Initialize the layer.

        Parameters:
            W: tuple[str, NDArray[np.floating]]
                Weights, with [name, weight array]. The shape of array is:
                    (filter_num, channel, filter_h, filter_w)
                    or (Fn, C, FH, FW)
            b: tuple[str, NDArray[np.floating]]
                Biases, with [name, array]. The shape of array is: (filter_num,)
            stride (int): Stride.
            pad (int): Padding.
        """
        self._W_filter = W
        self._b = b
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
            1. Input data x: (N, C, H, W).
                where:
                    N: batch size
                    C: channel
                    H: height
                    W: width
            2. im2col: (N * H_out * W_out, C * FH * FW).
                Where:
                    filter_size = (F_n, C, FH, FW)
                    H_out = (H - FH + 2 * pad) // stride + 1
                    W_out = (W - FW + 2 * pad) // stride + 1
            3. Filter matrix: (Fn, C * FH * FW).
            4. Perform matrix multiplication: (N * H_out * W_out, Fn)
                (N * H_out * W_out, C * FH * FW) * (Fn, C * FH * FW)'
            5. Add the bias (1, Fn).
            6. Reshape the output to: (N, Fn, H_out, W_out).

        Parameters:
            x: NDArray[np.floating]
                Input data. The shape is assumed to be a 4D array:
                    (batch_size, channel, height, width).

        Returns:
            NDArray[np.floating]: Output data, with shape:
                (C_out, filter_num, height, width).
        """
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Backward Propagation Dimension Process for CNN

        dout: Gradient of loss w.r.t. output (N, F_n, H_out, W_out)

        1. Gradient w.r.t. Bias (db):
            - Input: dout (N, F_n, H_out, W_out)
            - Process: Sum across batch (N), height (H_out), and width (W_out)
            - Output: db (F_n,)

        2. Gradient w.r.t. Weights (dW):
            - Input:
                - im2col result: (N * H_out * W_out, C * FH * FW)
                - dout reshaped to: (N * H_out * W_out, F_n)
            - Process:
                - Matrix multiplication: im2col.T * reshaped_dout
            - Output: dW (F_n, C * FH * FW)
            - Reshape to: dW (F_n, C, FH, FW)

        3. Gradient w.r.t. Input (dX):
            - Input:
                - dout reshaped to: (N * H_out * W_out, F_n)
                - Filter weights reshaped to: (F_n, C * FH * FW)
            - Process:
                - Matrix multiplication: reshaped_dout * reshaped_weights
                - Intermediate result: (N * H_out * W_out, C * FH * FW)
                - Apply col2im to reshape back to input dimensions
            - Output: dX (N, C, H, W)

        Summary of Dimensions:
            - db: (F_n,)
            - dW: (F_n, C, FH, FW)
            - dX: (N, C, H, W)

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
