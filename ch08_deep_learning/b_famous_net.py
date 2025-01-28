from common.layer_config import SequentialConfig


def le_net_config() -> SequentialConfig:
    """Configuration for LeNet model.

    Adjused: use the ReLU activation function and Max Pooling layer.

    Structural diagram:
        input: (1, 28, 28) with mnist dataset
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
    raise NotImplementedError


def alex_net_config() -> SequentialConfig:
    """Configuration for AlexNet model.

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
    raise NotImplementedError
