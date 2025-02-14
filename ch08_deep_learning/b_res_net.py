from common.layer_config import (
    AffineConfig,
    BottleneckBlocksConfig,
    Conv2dConfig,
    FlattenConfig,
    GlobalAvgPoolingConfig,
    MaxPool2dConfig,
    ReLUConfig,
    ResBlocksConfig,
    SequentialConfig,
)


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
    return SequentialConfig(
        hidden_layer_configs=(
            Conv2dConfig(
                in_channels=3,
                out_channels=64,
                param_suffix="1",
                kernel_size=(7, 7),
                stride=2,
                pad=3,
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(3, 3), stride=2, pad=0),
            ResBlocksConfig(
                in_channel=64,
                out_channel=64,
                stride=1,
                layer=2,
                param_suffix="2",
            ),
            ResBlocksConfig(
                in_channel=64,
                out_channel=128,
                stride=2,
                layer=2,
                param_suffix="3",
            ),
            ResBlocksConfig(
                in_channel=128,
                out_channel=256,
                stride=2,
                layer=2,
                param_suffix="4",
            ),
            ResBlocksConfig(
                in_channel=256,
                out_channel=512,
                stride=2,
                layer=2,
                param_suffix="5",
            ),
            GlobalAvgPoolingConfig(),
            FlattenConfig(),
            AffineConfig(in_size=512, out_size=1000, param_suffix="6"),
        )
    )


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
    return SequentialConfig(
        hidden_layer_configs=(
            Conv2dConfig(
                in_channels=3,
                out_channels=64,
                param_suffix="1",
                kernel_size=(7, 7),
                stride=2,
                pad=3,
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(3, 3), stride=2, pad=0),
            BottleneckBlocksConfig(
                in_channel=64,
                bottle_channel=64,
                out_channel=256,
                stride=1,
                layer=3,
                param_suffix="2",
            ),
            # 56x56 -> 28x28
            BottleneckBlocksConfig(
                in_channel=256,
                bottle_channel=128,
                out_channel=512,
                stride=2,
                layer=4,
                param_suffix="3",
            ),
            # 28x28 -> 14x14
            BottleneckBlocksConfig(
                in_channel=512,
                bottle_channel=256,
                out_channel=1024,
                stride=2,
                layer=6,
                param_suffix="4",
            ),
            # 14x14 -> 7x7
            BottleneckBlocksConfig(
                in_channel=1024,
                bottle_channel=512,
                out_channel=2048,
                stride=2,
                layer=3,
                param_suffix="5",
            ),
            # 7x7 -> 1x1
            GlobalAvgPoolingConfig(),
            FlattenConfig(),
            AffineConfig(in_size=512, out_size=1000, param_suffix="6"),
        )
    )
