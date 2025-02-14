import pytest

from common.default_type_array import np_randn
from common.layer_config import (
    BottleneckBlockConfig,
    BottleneckBlocksConfig,
    Conv2dGroupConfig,
    GlobalAvgPoolingConfig,
    ResBlockConfig,
    ResBlocksConfig,
)


class TestConv2dGroup:
    @pytest.mark.parametrize(
        "in_channels, out_channels, kernel_size, stride, pad, group, input_shape, expected_output_shape",
        [
            # Test with two groups
            (64, 128, (3, 3), 1, 1, 2, (1, 64, 32, 32), (1, 128, 32, 32)),
            # Test with stride=2 and downsampling
            (64, 128, (3, 3), 2, 1, 2, (1, 64, 32, 32), (1, 128, 16, 16)),
        ],
    )
    def test_conv2d_group_forward(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int,
        pad: int,
        group: int,
        input_shape: tuple[int, ...],
        expected_output_shape: tuple[int, ...],
    ) -> None:
        # Create Conv2dGroup layer
        conv_group_config = Conv2dGroupConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            group=group,
            param_suffix="1",
            kernel_size=kernel_size,
            stride=stride,
            pad=pad,
        )
        conv_group = conv_group_config.create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        output = conv_group.forward(x)
        # Check if the output shape matches the expected shape
        assert output.shape == expected_output_shape

    @pytest.mark.parametrize(
        "in_channels, out_channels, kernel_size, stride, pad, group, input_shape",
        [
            (64, 128, (3, 3), 1, 1, 2, (1, 64, 32, 32)),
        ],
    )
    def test_conv2d_group_backward(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int,
        pad: int,
        group: int,
        input_shape: tuple[int, ...],
    ) -> None:
        # Create Conv2dGroup layer
        conv_group_config = Conv2dGroupConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            group=group,
            param_suffix="1",
            kernel_size=kernel_size,
            stride=stride,
            pad=pad,
        )
        conv_group = conv_group_config.create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        output = conv_group.forward(x)

        # Backward pass
        dout = np_randn(output.shape)
        dx = conv_group.backward(dout)
        # Ensure gradients are computed and check the shape of the gradients
        assert dx.shape == x.shape

        # Get gradients
        grads = conv_group.param_grads()
        # Check that gradients for each parameter exist
        assert isinstance(grads, dict)
        assert len(grads) > 0  # There should be some parameters with gradients


class TestGlobalAvgPooling:
    @pytest.mark.parametrize(
        "input_shape, expected_output_shape",
        [
            # Example where the feature map size is 7x7
            ((1, 256, 7, 7), (1, 256, 1, 1)),
            # Example where the feature map size is 14x14
            ((1, 128, 14, 14), (1, 128, 1, 1)),
        ],
    )
    def test_global_avg_pooling_forward(
        self,
        input_shape: tuple[int, ...],
        expected_output_shape: tuple[int, ...],
    ) -> None:
        # Create the Global Average Pooling layer
        gap_layer = GlobalAvgPoolingConfig().create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)

        # Forward pass
        output = gap_layer.forward(x)

        # Check if the output shape matches the expected shape
        assert output.shape == expected_output_shape

    @pytest.mark.parametrize(
        "input_shape, dout_shape",
        [
            (
                (1, 256, 7, 7),
                (1, 256, 1, 1),
            ),  # Example where the feature map size is 7x7
            (
                (1, 128, 14, 14),
                (1, 128, 1, 1),
            ),  # Example where the feature map size is 14x14
        ],
    )
    def test_global_avg_pooling_backward(
        self, input_shape: tuple[int, ...], dout_shape: tuple[int, ...]
    ) -> None:
        # Create the Global Average Pooling layer
        gap_layer = GlobalAvgPoolingConfig().create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        gap_layer.forward(x)

        # Generate a random dout with the appropriate shape
        dout = np_randn(dout_shape)
        # Backward pass
        dx = gap_layer.backward(dout)
        # Ensure the gradients are of the correct shape
        assert dx.shape == x.shape

        # Get gradients
        grads = gap_layer.param_grads()
        # Check that gradients for each parameter exist
        assert isinstance(grads, dict)
        assert len(grads) == 0


class TestResBlock:
    @pytest.mark.parametrize(
        "input_shape, out_channels, expected_output_shape",
        [
            # No change in spatial dimensions for shallow resblock
            ((1, 64, 56, 56), 64, (1, 64, 56, 56)),
        ],
    )
    def test_resblock_forward(
        self,
        input_shape: tuple[int, ...],
        out_channels: int,
        expected_output_shape: tuple[int, ...],
    ) -> None:
        # Create mock Conv2d and BatchNorm layers for the ResBlock
        res_block = ResBlockConfig(
            in_channel=input_shape[1],
            out_channel=out_channels,
            stride=1,
            param_suffix="1",
        ).create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        output = res_block.forward(x)
        # Check if the output shape matches the expected shape
        assert output.shape == expected_output_shape

    @pytest.mark.parametrize(
        "input_shape, out_channel, stride",
        [
            ((1, 64, 56, 56), 32, 2),
            ((1, 32, 14, 14), 16, 2),
        ],
    )
    def test_resblock_backward(
        self, input_shape: tuple[int, ...], out_channel: int, stride: int
    ) -> None:
        # Create mock Conv2d and BatchNorm layers for the ResBlock
        res_block = ResBlockConfig(
            in_channel=input_shape[1],
            out_channel=out_channel,
            stride=stride,
            param_suffix="1",
        ).create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        output = res_block.forward(x)

        # Generate a random dout with the appropriate shape
        dout = np_randn(output.shape)
        # Backward pass
        dx = res_block.backward(dout)
        # Ensure the gradients are of the correct shape
        assert dx.shape == x.shape

        # Get gradients
        grads = res_block.param_grads()
        # Check that gradients for each parameter exist
        assert isinstance(grads, dict)
        assert len(grads) > 0  # There should be some parameters with gradients


@pytest.mark.parametrize(
    "in_shape, in_channel, out_channel, stride, layer, out_shape",
    [
        # Single layer
        ((1, 64, 32, 32), 64, 128, 1, 1, (1, 128, 32, 32)),
        # Multiple layers with stride
        ((1, 64, 32, 32), 64, 256, 2, 2, (1, 256, 16, 16)),
    ],
)
def test_res_blocks_config_and_layers(
    in_shape: tuple[int, ...],
    in_channel: int,
    out_channel: int,
    stride: int,
    layer: int,
    out_shape: tuple[int, ...],
) -> None:
    config = ResBlocksConfig(
        in_channel=in_channel,
        out_channel=out_channel,
        stride=stride,
        layer=layer,
        param_suffix="1",
    )
    res_blocks = config.create()

    # Test forward/backward passes
    x = np_randn(in_shape)
    output = res_blocks.forward(x)
    assert output.shape == out_shape

    grad = np_randn(out_shape)
    dx = res_blocks.backward(grad)
    assert dx.shape == in_shape

    # Verify parameters
    params = res_blocks.named_params()
    for idx in range(layer):
        suffix = f"{config.param_suffix}_{idx + 1}"
        assert any(suffix in key for key in params.keys())


class TestBottleneckBlock:
    @pytest.mark.parametrize(
        "input_shape, bottle_channel, stride, expected_output_shape",
        [
            ((1, 64, 56, 56), 64, 1, (1, 256, 56, 56)),
            ((1, 128, 28, 28), 64, 2, (1, 256, 14, 14)),
        ],
    )
    def test_bottleneck_forward(
        self,
        input_shape: tuple[int, ...],
        bottle_channel: int,
        stride: int,
        expected_output_shape: tuple[int, ...],
    ) -> None:
        # Create the BottleneckBlock
        bottleneck_block = BottleneckBlockConfig(
            in_channel=input_shape[1],
            bottle_channel=bottle_channel,
            out_channel=expected_output_shape[1],
            stride=stride,
            param_suffix="1",
        ).create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        output = bottleneck_block.forward(x)
        # Check if the output shape matches the expected shape
        assert output.shape == expected_output_shape

    @pytest.mark.parametrize(
        "input_shape, bottle_channel, out_channel, stride",
        [
            ((1, 64, 56, 56), 64, 128, 1),
            ((1, 128, 28, 28), 64, 256, 2),
        ],
    )
    def test_bottleneck_backward(
        self,
        input_shape: tuple[int, ...],
        bottle_channel: int,
        out_channel: int,
        stride: int,
    ) -> None:
        # Create the BottleneckBlock
        bottleneck_block = BottleneckBlockConfig(
            in_channel=input_shape[1],
            bottle_channel=bottle_channel,
            out_channel=out_channel,
            stride=stride,
            param_suffix="1",
        ).create()

        # Generate random input data with the given input shape
        x = np_randn(input_shape)
        # Forward pass
        output = bottleneck_block.forward(x)

        # Generate random dout with the appropriate shape
        dout = np_randn(output.shape)
        # Backward pass
        dx = bottleneck_block.backward(dout)
        # Ensure the gradients are of the correct shape
        assert dx.shape == x.shape

        # Get gradients
        grads = bottleneck_block.param_grads()
        # Check that gradients for each parameter exist
        assert isinstance(grads, dict)
        assert len(grads) > 0  # There should be some parameters with gradients


@pytest.mark.parametrize(
    "in_shape, in_channel, bottle_channel, out_channel, stride, layer, out_shape",
    [
        # Single bottleneck block
        ((1, 64, 56, 56), 64, 64, 256, 1, 1, (1, 256, 56, 56)),
        # Multiple bottleneck blocks with stride
        ((1, 256, 56, 56), 256, 128, 512, 2, 3, (1, 512, 28, 28)),
    ],
)
def test_bottleneck_blocks_config_and_layers(
    in_shape: tuple[int, ...],
    in_channel: int,
    bottle_channel: int,
    out_channel: int,
    stride: int,
    layer: int,
    out_shape: tuple[int, ...],
) -> None:
    config = BottleneckBlocksConfig(
        in_channel=in_channel,
        bottle_channel=bottle_channel,
        out_channel=out_channel,
        stride=stride,
        layer=layer,
        param_suffix="1",
    )
    bottleneck_blocks = config.create()

    # Test forward pass
    x = np_randn(in_shape)
    output = bottleneck_blocks.forward(x)
    assert output.shape == out_shape

    # Test backward pass
    dout = np_randn(out_shape)
    dx = bottleneck_blocks.backward(dout)
    assert dx.shape == in_shape

    # Verify parameters and gradients
    params = bottleneck_blocks.named_params()
    grads = bottleneck_blocks.param_grads()

    for idx in range(layer):
        suffix = f"{config.param_suffix}_{idx + 1}"
        assert any(suffix in key for key in params.keys())

    assert isinstance(grads, dict)
    assert len(grads) > 0
