from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ch05_backpropagation.b_layer import ReLU
from ch07_cnn.c_convolution_layer import Conv2d
from common.base import Layer
from common.default_type_array import np_float


class Conv2dGroup(Layer):
    """Conv2dGroup is a custom layer that groups multiple Conv2d layers and
    processes different input channels through each Conv2d layer in parallel.

    Diagram:
        Input: (N, C, H, W)
        ├── Conv2d (group 1)
        │   Input: (N, C/G, H, W)
        │   Output: (N, C_out1, H_out, W_out)
        ├── Conv2d (group 2)
        │   Input: (N, C/G, H, W)
        │   Output: (N, C_out2, H_out, W_out)
        ├── ...
        └── Conv2d (group G)
            Input: (N, C/G, H, W)
            Output: (N, C_outG, H_out, W_out)
        Output: (N, C_out, H_out, W_out)
            where C_out = C_out1 + ... + C_outG

    Why Use Grouped Convolution?
        1. Computational Efficiency:
            Grouped convolutions reduce the number of operations compared to a
            full convolution.
        2. Multi-GPU Parallelism (Original AlexNet):
            In the original AlexNet, each group was processed on a different
            GPU to utilize multiple GPUs effectively.
        3. Better Representation Learning:
            By keeping groups separate, some architectures (like MobileNet) can
            improve feature extraction and efficiency.
    """

    def __init__(self, conv_layers: Sequence[Conv2d]) -> None:
        self._conv_layers = conv_layers
        self._group = len(conv_layers)
        self._params = {
            key: value
            for layer in self._conv_layers
            for key, value in layer.named_params().items()
        }

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return self._params

    def train(self, flag: bool) -> None:
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        _, c, _, _ = x.shape
        assert c % self._group == 0
        channel_evry_group = int(c / self._group)

        y = {}
        for idx, layer in enumerate(self._conv_layers):
            start_c = idx * channel_evry_group
            end_c = start_c + channel_evry_group
            # TODO: use multi-prosses to paralle it
            y[idx] = layer.forward(x[:, range(start_c, end_c), :, :])

        return np.concatenate([y[idx] for idx in range(self._group)], axis=1)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        _, c_out, _, _ = dout.shape
        assert c_out % self._group == 0
        channel_evry_group = int(c_out / self._group)

        dx = {}
        for idx, layer in enumerate(self._conv_layers):
            start_c = idx * channel_evry_group
            end_c = start_c + channel_evry_group
            # TODO: use multi-prosses to paralle it
            dx[idx] = layer.backward(dout[:, range(start_c, end_c), :, :])

        return np.concatenate([dx[idx] for idx in range(self._group)], axis=1)

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        # Use a dictionary comprehension to combine all gradients in one step
        return {
            key: value
            for layer in self._conv_layers
            for key, value in layer.param_grads().items()
        }


class GlobalAvgPooling(Layer):
    """Global Average Pooling computes the average of each feature map over
    the entire spatial dimensions (height and width).

    Input feature map: N, C, H, W
    Output after GAP: N, C, 1, 1
    """

    def __init__(self) -> None:
        self._x_h: int | None = None
        self._x_w: int | None = None

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        return {}

    def train(self, flag: bool) -> None:
        pass

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        _, _, h, w = x.shape
        self._x_h, self._x_w = h, w
        res: NDArray[np.floating] = np.sum(
            x, axis=(2, 3), keepdims=True
        ) / np_float(h * w)
        return res

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        assert self._x_h is not None
        assert self._x_w is not None

        _, _, h_out, w_out = dout.shape
        assert h_out == w_out == 1

        return np.tile(dout, (1, 1, self._x_h, self._x_w)) * np_float(
            self._x_h * self._x_w
        )

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        # Use a dictionary comprehension to combine all gradients in one step
        return {}


class ResBlock(Layer):
    """Basic residual block used in shallow ResNet architectures.

    This block consists of two convolutional layers with a skip connection,
    enabling the network to learn residual functions.

    Diagram:
        input -> Conv1 (3x3) -> BatchNorm -> ReLU ->
                Conv2 (3x3) -> BatchNorm -> + -> ReLU -> output
          |_________________________________|

    Each convolutional layer is followed by Batch Normalization (`BatchNorm`)
    before applying ReLU activation. The skip connection adds the original
    input to the transformed output before the final activation.

    **Use Cases:**
        - Used in **shallower ResNet architectures** such as **ResNet-18 and ResNet-34**.
        - Suitable for **smaller datasets** or when computational efficiency is a priority.
        - Preferred when working with **low to medium depth networks**.
        - Used when model depth is **not too deep**, so no need for a bottleneck structure.

    Attributes:
        conv1 (Conv2D): 3x3 convolution for feature extraction.
        bn1 (BatchNorm2D): Batch Normalization for conv1.
        conv2 (Conv2D): 3x3 convolution for feature extraction.
        bn2 (BatchNorm2D): Batch Normalization for conv2.
        shortcut (Sequential or Identity): Skip connection for residual learning.
        relu (ReLU): Activation function applied after convolutions.
    """

    def __init__(
        self,
        conv1: Layer,
        conv2: Layer,
        batch_norm1: Layer,
        batch_norm2: Layer,
        shortcut: Sequence[Layer],
    ) -> None:
        self._first_5_layers = (
            conv1,
            batch_norm1,
            ReLU(inplace=True),
            conv2,
            batch_norm2,
        )
        self._relu2 = ReLU(inplace=True)
        # Shortcut (skip connection) to match the dimensions of the output
        # If stride=2, we need a 1x1 convolution to match the spatial dimensions
        self._shortcut = tuple(shortcut)

        self._params = {
            key: value
            for layer in (self._first_5_layers + self._shortcut)
            for key, value in layer.named_params().items()
        }

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network."""
        return self._params

    def train(self, flag: bool) -> None:
        """Set the training flag of the layer."""
        # the ReLU is not necessary to call train
        for layer in self._first_5_layers + self._shortcut:
            layer.train(flag)

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        temp_x = x
        for layer in self._first_5_layers:
            temp_x = layer.forward(temp_x)

        skip_x = x
        for layer in self._shortcut:
            skip_x = layer.forward(skip_x)

        return self._relu2.forward(skip_x + temp_x)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        d_relu2 = self._relu2.backward(dout)
        temp_grads = d_relu2
        for layer in reversed(self._first_5_layers):
            temp_grads = layer.backward(temp_grads)

        skip_grads = d_relu2
        for layer in reversed(self._shortcut):
            skip_grads = layer.backward(skip_grads)

        return temp_grads + skip_grads

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters.

        This have to be called after the backward pass.
        """
        return {
            key: value
            for layer in (self._first_5_layers + self._shortcut)
            for key, value in layer.param_grads().items()
        }


class BottleneckBlock(Layer):
    """Bottleneck block used in deep ResNet architectures.

    This block consists of three convolutional layers with a skip connection.
    It follows the bottleneck design to reduce computation while maintaining
    performance.

    Diagram:
        input -> Conv1 (1x1, reduce channels) -> BatchNorm -> ReLU ->
                Conv2 (3x3, main conv) -> BatchNorm -> ReLU ->
                Conv3 (1x1, restore channels) -> BatchNorm -> + -> ReLU -> output
          |___________________________________________________|

    The first 1x1 convolution reduces dimensionality,
    the 3x3 convolution performs the main feature extraction,
    and the final 1x1 convolution restores the original channel size.
    Batch Normalization (`BatchNorm`) is applied after each convolution
    before ReLU activation to stabilize training and improve performance.

    The skip connection adds the original input to the transformed output
    before applying the final ReLU activation.

    **Use Cases:**
        - Used in **deeper ResNet architectures** such as **ResNet-50, ResNet-101, and ResNet-152**.
        - Suitable for **large-scale image classification** tasks (e.g., ImageNet).
        - Helps **reduce computational cost** while maintaining high model capacity.
        - Preferred when working with **deep neural networks** where efficiency is critical.

    Attributes:
        conv1 (Conv2D): 1x1 convolution to reduce the number of channels.
        bn1 (BatchNorm2D): Batch Normalization for conv1.
        conv2 (Conv2D): 3x3 convolution for main feature extraction.
        bn2 (BatchNorm2D): Batch Normalization for conv2.
        conv3 (Conv2D): 1x1 convolution to restore the number of channels.
        bn3 (BatchNorm2D): Batch Normalization for conv3.
        shortcut (Sequential or Identity): Skip connection for residual learning.
        relu (ReLU): Activation function applied after each convolution.
    """

    def __init__(
        self,
        conv1: Layer,
        conv2: Layer,
        conv3: Layer,
        batch_norm1: Layer,
        batch_norm2: Layer,
        batch_norm3: Layer,
        shortcut: Sequence[Layer],
    ) -> None:
        self._first_8_layers = (
            conv1,
            batch_norm1,
            ReLU(inplace=True),
            conv2,
            batch_norm2,
            ReLU(inplace=True),
            conv3,
            batch_norm3,
        )
        self._shortcut = tuple(shortcut)
        self._relu3 = ReLU()

        self._params = {
            key: value
            for layer in (self._first_8_layers + self._shortcut)
            for key, value in layer.named_params().items()
        }

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network."""
        return self._params

    def train(self, flag: bool) -> None:
        """Set the training flag of the layer."""
        # the ReLU is not necesasry to call train
        for layer in self._first_8_layers + self._shortcut:
            layer.train(flag)

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        temp_x = x
        for layer in self._first_8_layers:
            temp_x = layer.forward(temp_x)

        skip_x = x
        for layer in self._shortcut:
            skip_x = layer.forward(skip_x)

        return self._relu3.forward(temp_x + skip_x)

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        d_relu3 = self._relu3.backward(dout)
        temp_grads = d_relu3
        for layer in reversed(self._first_8_layers):
            temp_grads = layer.backward(temp_grads)

        skip_grad = d_relu3
        for layer in reversed(self._shortcut):
            skip_grad = layer.backward(skip_grad)

        return temp_grads + skip_grad

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters.

        This have to be called after the backward pass.
        """
        return {
            key: value
            for layer in (self._first_8_layers + self._shortcut)
            for key, value in layer.param_grads().items()
        }
