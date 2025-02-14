from common.layer_config import (
    AffineConfig,
    Conv2dConfig,
    DropoutConfig,
    FlattenConfig,
    MaxPool2dConfig,
    ReLUConfig,
    SequentialConfig,
)


def deep_2d_net_config() -> SequentialConfig:
    """Configuration for a deep 2D convolutional neural network.


    Structural diagram:
        input: (1, 28, 28) with mnist dataset
        Layer 1: Convolutional Layer
            - Filters: 16
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (16, 28, 28)

        Layer 2: Convolutional Layer
            - Filters: 16
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (16, 28, 28)

        Layer 3: Max Pooling Layer
            - Pool size: (2, 2)
            - Stride: 2
            - Output: (16, 14, 14)

        Layer 4: Convolutional Layer
            - Filters: 32
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (32, 14, 14)

        Layer 5: Convolutional Layer
            - Filters: 32
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (32, 14, 14)

        Layer 6: Max Pooling Layer
            - Pool size: (2, 2)
            - Stride: 2
            - Output: (32, 7, 7)

        Layer 7: Convolutional Layer
            - Filters: 64
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (64, 7, 7)

        Layer 8: Convolutional Layer
            - Filters: 64
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (64, 7, 7)

        Layer 9: Max Pooling Layer
            - Pool size: (2, 2)
            - Stride: 2
            - Output: (64, 3, 3)

        Layer 10: Fully Connected Layer
            - Neurons: 50
            - Activation: ReLU
            - Output: (50,)

        Layer 11: Dropout Layer
            - Dropout ratio: 0.5

        Layer 11: Fully Connected Layer
            - Neurons: 10
            - Activation:
            - Output: (10,)

        layer 12: Dropout Layer
            - Dropout ratio: 0.5
    """
    return SequentialConfig(
        hidden_layer_configs=(
            Conv2dConfig(
                in_channels=1,
                out_channels=16,
                param_suffix="1",
                kernel_size=(3, 3),
                stride=1,
                pad=1,
            ),
            ReLUConfig(),
            Conv2dConfig(
                in_channels=16,
                out_channels=16,
                param_suffix="2",
                kernel_size=(3, 3),
                stride=1,
                pad=1,
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2, pad=0),
            Conv2dConfig(
                in_channels=16,
                out_channels=32,
                param_suffix="3",
                kernel_size=(3, 3),
                stride=1,
                pad=1,
            ),
            ReLUConfig(),
            Conv2dConfig(
                in_channels=32,
                out_channels=32,
                param_suffix="4",
                kernel_size=(3, 3),
                stride=1,
                pad=1,
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2, pad=0),
            Conv2dConfig(
                in_channels=32,
                out_channels=64,
                param_suffix="5",
                kernel_size=(3, 3),
                stride=1,
                pad=1,
            ),
            ReLUConfig(),
            Conv2dConfig(
                in_channels=64,
                out_channels=64,
                param_suffix="6",
                kernel_size=(3, 3),
                stride=1,
                pad=1,
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2, pad=0),
            FlattenConfig(),
            AffineConfig(in_size=64 * 3 * 3, out_size=50, param_suffix="7"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=0.5),
            AffineConfig(in_size=50, out_size=10, param_suffix="8"),
            DropoutConfig(dropout_ratio=0.5),
        )
    )
