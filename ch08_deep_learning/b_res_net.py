import numpy as np
from numpy.typing import NDArray

from common.base import Layer
from common.layer_config import SequentialConfig


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

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network."""
        raise NotImplementedError

    def train(self, flag: bool) -> None:
        """Set the training flag of the layer."""
        raise NotImplementedError

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters.

        This have to be called after the backward pass.
        """
        raise NotImplementedError


class ResBottleneckBlock(Layer):
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

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network."""
        raise NotImplementedError

    def train(self, flag: bool) -> None:
        """Set the training flag of the layer."""
        raise NotImplementedError

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters.

        This have to be called after the backward pass.
        """
        raise NotImplementedError


def res_net_18_config() -> SequentialConfig:
    """Configuration for ResNet model.

    This implementation is based on the original ResNet paper:
    "Deep Residual Learning for Image Recognition" by Kaiming He et al.

    The architecture consists of a stack of residual blocks. Each block
    has a skip connection to the output, which helps to avoid the vanishing
    gradient problem.

    Layers:
    1. Input: 224x224x3 (RGB image)
    2. Conv1: Convolutional layer
        - Filters: 64
        - Kernel size: 7x7
        - Stride: 2
        - Padding: 3
        - Output: 112x112x64
        - Activation: ReLU
    3. Pool1: Max Pooling
        - Kernel size: 3x3
        - Stride: 2
        - Output: 56x56x64
    4. ResBlock1: Residual Block
        - Layers: 2
        - Filters: 64
        - Output: 56x56x64
    5. ResBlock2: Residual Block
        - Layers: 2
        - Filters: 128
        - Stride: 2
        - Output: 28x28x128
    6. ResBlock3: Residual Block
        - Layers: 2
        - Filters: 256
        - Stride: 2
        - Output: 14x14x256
    7. ResBlock4: Residual Block
        - Layers: 2
        - Filters: 512
        - Stride: 2
        - Output: 7x7x512
    8. Pool2: Global Average Pooling
        - Output: 1x1x512
    9. FC: Fully Connected Layer
        - Neurons: 1000 (number of classes in ImageNet)
        - Activation: Softmax
    """
    raise NotImplementedError


def res_net_50_config() -> SequentialConfig:
    """Configuration for ResNet-50 model.

    This implementation is based on the original ResNet paper:
    "Deep Residual Learning for Image Recognition" by Kaiming He et al.

    The architecture consists of a stack of residual blocks. Each block
    has a skip connection to the output, which helps to avoid the vanishing
    gradient problem.

    The ResNet-50 model uses bottleneck blocks to reduce computation while
    maintaining high performance. The architecture is designed to process
    images of size 224x224x3 (RGB channels).

    Layers:
    1. Input: 224x224x3 (RGB image)
    2. Conv1: Convolutional layer
        - Filters: 64
        - Kernel size: 7x7
        - Stride: 2
        - Padding: 3
        - Output: 112x112x64
        - Activation: ReLU
    3. Pool1: Max Pooling
        - Kernel size: 3x3
        - Stride: 2
        - Output: 56x56x64
    4. ResBlock1: Bottleneck Block
        - Layers: 3
        - Filters: 64, 64, 256
        - Output: 56x56x256
    5. ResBlock2: Bottleneck Block
        - Layers: 4
        - Filters: 128, 128, 512
        - Stride: 2
        - Output: 28x28x512
    6. ResBlock3: Bottleneck Block
        - Layers: 6
        - Filters: 256, 256, 1024
        - Stride: 2
        - Output: 14x14x1024
    7. ResBlock4: Bottleneck Block
        - Layers: 3
        - Filters: 512, 512, 2048
        - Stride: 2
        - Output: 7x7x2048
    8. Pool2: Global Average Pooling
        - Output: 1x1x2048
    9. FC: Fully Connected Layer
        - Neurons: 1000 (number of classes in ImageNet)
        - Activation: Softmax
    """
    raise NotImplementedError
