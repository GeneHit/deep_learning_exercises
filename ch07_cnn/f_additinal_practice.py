import numpy as np
from numpy.typing import NDArray

from common.base import NueralNet


class LeNet(NueralNet):
    """LeNet model for MNIST dataset.

    Adjused: use the ReLU activation function and Max Pooling layer.

    Structural diagram:

    Input: (1, 28, 28)

    Layer 1: Convolutional Layer
        - Filters: 6
        - Filter size: (5, 5)
        - Stride: 1
        - Padding: 0
        - Activation: ReLU
        - Output: (6, 24, 24)

    Layer 2: Max Pooling Layer
        - Pool size: (2, 2)
        - Stride: 2
        - Output: (6, 12, 12)

    Layer 3: Convolutional Layer
        - Filters: 16
        - Filter size: (5, 5)
        - Stride: 1
        - Padding: 0
        - Activation: ReLU
        - Output: (16, 8, 8)

    Layer 4: Max Pooling Layer
        - Pool size: (2, 2)
        - Stride: 2
        - Output: (16, 4, 4)

    Layer 5: Fully Connected Layer
        - Neurons: 120
        - Activation: ReLU
        - Output: (120,)

    Layer 6: Fully Connected Layer
        - Neurons: 84
        - Activation: ReLU
        - Output: (84,)

    Layer 7: Fully Connected Layer
        - Neurons: 10
        - Activation: Softmax
        - Output: (10,)
    """

    def __init__(self) -> None:
        """Initialize the LeNet model."""
        self.params = {}
        self.params["W1"] = np.random.randn(6, 1, 5, 5) * np.sqrt(
            2.0 / (1 * 5 * 5)
        )
        self.params["b1"] = np.zeros(6)
        self.params["W2"] = np.random.randn(16, 6, 5, 5) * np.sqrt(
            2.0 / (6 * 5 * 5)
        )
        self.params["b2"] = np.zeros(16)
        self.params["W3"] = np.random.randn(120, 16 * 4 * 4) * np.sqrt(
            2.0 / (16 * 4 * 4)
        )
        self.params["b3"] = np.zeros(120)
        self.params["W4"] = np.random.randn(84, 120) * np.sqrt(2.0 / 120)
        self.params["b4"] = np.zeros(84)
        self.params["W5"] = np.random.randn(10, 84) * np.sqrt(2.0 / 84)
        self.params["b5"] = np.zeros(10)

    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """
        raise NotImplementedError

    def predict(
        self, x: NDArray[np.floating], train_flag: bool = False
    ) -> NDArray[np.floating]:
        """See the base class."""
        raise NotImplementedError

    def loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        train_flag: bool = False,
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def accuracy(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        raise NotImplementedError


class AlexNet(NueralNet):
    """AlexNet model.

    This implementation includes the normalization and dropout layers,
    which are key components of the original AlexNet. The architecture
    is designed to process images of size 227x227x3 (RGB channels).

    Notice: The original AlexNet used Local Response Normalization, but
    this implementation uses Batch Normalization. BatchNorm is more
    effective and easier to implement.

    Layers:
    1. Input: 227x227x3 (RGB image)
    2. Conv1: Convolutional layer
        - Filters: 96
        - Kernel size: 11x11
        - Stride: 4
        - Padding: 0
        - Output: 55x55x96
        - Activation: ReLU
    3. Norm1: BatchNorm (original paper used Local Response Normalization)
    4. Pool1: Max Pooling
        - Kernel size: 3x3
        - Stride: 2
        - Output: 27x27x96
    5. Conv2: Convolutional layer
        - Filters: 256
        - Kernel size: 5x5
        - Stride: 1
        - Padding: 2
        - Groups: 2 (split across GPUs in original paper)
        - Output: 27x27x256
        - Activation: ReLU
    6. Norm2: BatchNorm (original paper used Local Response Normalization)
    7. Pool2: Max Pooling
        - Kernel size: 3x3
        - Stride: 2
        - Output: 13x13x256
    8. Conv3: Convolutional layer
        - Filters: 384
        - Kernel size: 3x3
        - Stride: 1
        - Padding: 1
        - Output: 13x13x384
        - Activation: ReLU
    9. Conv4: Convolutional layer
        - Filters: 384
        - Kernel size: 3x3
        - Stride: 1
        - Padding: 1
        - Groups: 2
        - Output: 13x13x384
        - Activation: ReLU
    10. Conv5: Convolutional layer
        - Filters: 256
        - Kernel size: 3x3
        - Stride: 1
        - Padding: 1
        - Groups: 2
        - Output: 13x13x256
        - Activation: ReLU
    11. Pool3: Max Pooling
        - Kernel size: 3x3
        - Stride: 2
        - Output: 6x6x256
    12. Flatten: Flatten the tensor into a vector
        - Output: 6x6x256 = 9216
    13. FC1: Fully Connected Layer
        - Input: 9216
        - Neurons: 4096
        - Activation: ReLU
        - Dropout: Dropout with probability 0.5
    14. FC2: Fully Connected Layer
        - Input: 4096
        - Neurons: 4096
        - Activation: ReLU
        - Dropout: Dropout with probability 0.5
    15. FC3 (Output Layer): Fully Connected Layer
        - Input: 4096
        - Neurons: 1000 (number of classes in ImageNet)
        - Activation: Softmax
    """

    def __init__(self) -> None:
        """Initialize the AlexNet model."""
        self.params = {}
        self.params["W1"] = np.random.randn(96, 3, 11, 11) * np.sqrt(
            2.0 / (3 * 11 * 11)
        )
        self.params["b1"] = np.zeros(96)
        self.params["W2"] = np.random.randn(256, 96, 5, 5) * np.sqrt(
            2.0 / (96 * 5 * 5)
        )
        self.params["b2"] = np.zeros(256)
        self.params["W3"] = np.random.randn(384, 256, 3, 3) * np.sqrt(
            2.0 / (256 * 3 * 3)
        )
        self.params["b3"] = np.zeros(384)
        self.params["W4"] = np.random.randn(384, 384, 3, 3) * np.sqrt(
            2.0 / (384 * 3 * 3)
        )
        self.params["b4"] = np.zeros(384)
        self.params["W5"] = np.random.randn(256, 384, 3, 3) * np.sqrt(
            2.0 / (384 * 3 * 3)
        )
        self.params["b5"] = np.zeros(256)
        self.params["W6"] = np.random.randn(4096, 256 * 6 * 6) * np.sqrt(
            2.0 / (256 * 6 * 6)
        )
        self.params["b6"] = np.zeros(4096)
        self.params["W7"] = np.random.randn(4096, 4096) * np.sqrt(2.0 / 4096)
        self.params["b7"] = np.zeros(4096)
        self.params["W8"] = np.random.randn(1000, 4096) * np.sqrt(2.0 / 4096)
        self.params["b8"] = np.zeros(1000)

    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """
        raise NotImplementedError

    def predict(
        self, x: NDArray[np.floating], train_flag: bool = False
    ) -> NDArray[np.floating]:
        """See the base class."""
        raise NotImplementedError

    def loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        train_flag: bool = False,
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def accuracy(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> float:
        """See the base class."""
        raise NotImplementedError

    def gradient(
        self, x: NDArray[np.floating], t: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        """See the base class."""
        raise NotImplementedError
