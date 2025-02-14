"""
This module defines configuration classes for various neural network layers.
Each configuration class:
    1. inherits from `LayerConfig`
    2. provides the necessary parameters
    3. provide create methods the corresponding layer instances.
Classes:
    AffineConfig: Configuration for the Affine layer.
    AvgPool2dConfig: Configuration for the 2D average pooling layer.
    BatchNorm1dConfig: Configuration for the BatchNorm1d layer.
    BatchNorm2dConfig: Configuration for the BatchNorm2d layer.
    Conv2dConfig: Configuration for the Convolution layer.
    DropoutConfig: Configuration for the Dropout layer.
    Dropout2dConfig: Configuration for the Dropout2d layer.
    FlattenConfig: Configuration for the Flatten layer.
    MaxPool2dConfig: Configuration for the 2D max pooling layer.
    ReLUConfig: Configuration for the ReLU layer.
    SequentialConfig: Configuration for the Sequential layer.
    SigmoidConfig: Configuration for the Sigmoid layer.
    SoftmaxConfig: Configuration for the Softmax layer.
    SoftmaxWithLossConfig: Configuration for the Softmax with loss layer.
Functions:
    create_layers: Creates a list of layers based on the provided configurations.
    assert_keys_if_params_provided: Asserts that the parameters are provided
        for the given keys.
"""

import pickle
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ch05_backpropagation.b_layer import (
    Affine,
    ReLU,
    Sigmoid,
    Softmax,
    SoftmaxWithLoss,
)
from ch06_learning_technique.b_weight_init import (
    generate_init_bias,
    generate_init_weight,
)
from ch06_learning_technique.c_batch_normalization import (
    BatchNorm1d,
    BatchNorm2d,
)
from ch06_learning_technique.d_reg_dropout import Dropout, Dropout2d
from ch06_learning_technique.d_reg_weight_decay import Sequential
from ch07_cnn.c_convolution_layer import Conv2d
from ch07_cnn.d_pooling_layer import AvgPool2d, Flatten, MaxPool2d
from ch08_deep_learning.b_layer_block import (
    BottleneckBlock,
    Conv2dGroup,
    GlobalAvgPooling,
    ResBlock,
)
from common.base import Layer, LayerConfig


@dataclass(frozen=True, kw_only=True)
class ParameterInitConfig:
    """Configuration for the parameter initialization."""

    initializer: str
    """The initializer for the weights of the layer.

    Options:
        - "he_normal": He normal initializer.
        - "he_uniform": He uniform initializer.
        - "xavier_normal": Xavier normal initializer.
        - "xavier_uniform": Xavier uniform initializer.
        - "normal": Normal distribution with weight_init_std.
        - "uniform": Uniform distribution with weight_init_std.
    """

    mode: str
    """The mode for the initialization.

    Options:
        - "fan_in": The number of input units in the weight tensor.
        - "fan_out": The number of output units in the weight tensor.
        - "fan_avg": The average of the number of input and output units.
    """

    weight_init_std: float | None = None
    """The standard deviation for the distribution of the weights.

    It is only used when the initializer is "normal" or "uniform".
    """

    def __post_init__(self) -> None:
        # checker
        if self.initializer in {"normal", "uniform"}:
            assert self.weight_init_std is not None, (
                "The weight_init_std have to be provided for the initializer."
            )


@dataclass(frozen=True, kw_only=True)
class AffineConfig(LayerConfig):
    """Configuration for the Affine layer."""

    in_size: int
    """The input size of the layer."""

    out_size: int
    """The output size of the layer."""

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in"
    )
    """The configuration for the parameter initialization."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        w_name = f"fc{self.param_suffix}_w"
        b_name = f"fc{self.param_suffix}_b"
        if parameters is not None:
            assert_keys_if_params_provided(parameters, [w_name, b_name])
            w = parameters[w_name]
            b = parameters[b_name]
        else:
            w = generate_init_weight(
                weight_shape=(self.in_size, self.out_size),
                initializer=self.param_init.initializer,
                mode=self.param_init.mode,
                stddev=self.param_init.weight_init_std,
            )
            b = generate_init_bias(
                bias_shape=(1, self.out_size), initializer="zeros"
            )

        return Affine(w=(w_name, w), b=(b_name, b))


@dataclass(frozen=True, kw_only=True)
class AvgPool2dConfig(LayerConfig):
    """Configuration for the 2D average pooling layer."""

    kernel_size: tuple[int, int]
    """The size of the pooling window."""

    stride: int = 1
    """The stride of the pooling window."""

    pad: int = 0
    """The padding size."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return AvgPool2d(
            kenel_size=self.kernel_size,
            stride=self.stride,
            pad=self.pad,
        )


@dataclass(frozen=True, kw_only=True)
class BatchNorm1dConfig(LayerConfig):
    """Configuration for the BatchNorm1d layer."""

    num_feature: int
    """The number of features in the input tensor."""

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `gamma_{param_suffix}` and `beta_{param_suffix}`.
    """

    momentum: float = 0.9
    """The momentum for the moving average."""

    affine: bool = True
    """If True, learnable affine parameters (gamma and beta) are used."""

    track_running_stats: bool = True
    """If True, running mean and variance are tracked during training."""

    eps: float = 1e-5
    """A small constant added to the denominator for numerical stability."""

    def _get_params(
        self,
        parameters: dict[str, NDArray[np.floating]] | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> tuple[
        tuple[str, NDArray[np.floating]],
        tuple[str, NDArray[np.floating]],
        tuple[str, NDArray[np.floating]],
        tuple[str, NDArray[np.floating]],
    ]:
        gamma_name = f"bn{self.param_suffix}_gamma"
        beta_name = f"bn{self.param_suffix}_beta"
        mean_name = f"bn{self.param_suffix}_running_mean"
        var_name = f"bn{self.param_suffix}_running_var"
        if parameters is not None:
            assert_keys_if_params_provided(
                parameters, [gamma_name, beta_name, mean_name, var_name]
            )
            gamma = parameters[gamma_name]
            beta = parameters[beta_name]
            running_mean = parameters[mean_name]
            running_var = parameters[var_name]
        else:
            assert shape is not None, "The shape has to be provided."
            gamma = generate_init_bias(bias_shape=shape, initializer="ones")
            beta = generate_init_bias(bias_shape=shape, initializer="zeros")
            running_mean = generate_init_bias(
                bias_shape=shape, initializer="zeros"
            )
            running_var = generate_init_bias(
                bias_shape=shape, initializer="ones"
            )

        return (
            (gamma_name, gamma),
            (beta_name, beta),
            (mean_name, running_mean),
            (var_name, running_var),
        )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        gamma, beta, mean, var = self._get_params(
            parameters, shape=(1, self.num_feature)
        )
        return BatchNorm1d(
            gamma=gamma,
            beta=beta,
            running_mean=mean,
            running_var=var,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            eps=self.eps,
        )


@dataclass(frozen=True, kw_only=True)
class BatchNorm2dConfig(BatchNorm1dConfig):
    """Configuration for the BatchNormal2d layer."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        gamma, beta, mean, var = self._get_params(
            parameters, shape=(1, self.num_feature, 1, 1)
        )
        return BatchNorm2d(
            gamma=gamma,
            beta=beta,
            running_mean=mean,
            running_var=var,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            eps=self.eps,
        )


@dataclass(frozen=True, kw_only=True)
class Conv2dConfig(LayerConfig):
    """Configuration for the Convolution layer."""

    in_channels: int
    """The number of input channels."""

    out_channels: int
    """The number of output channels."""

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    kernel_size: tuple[int, int]
    """The size of the kernel."""

    stride: int = 1
    """The stride of the kernel."""

    pad: int = 0
    """The padding size."""

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in", weight_init_std=None
    )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        w_name = f"conv{self.param_suffix}_w"
        b_name = f"conv{self.param_suffix}_b"
        if parameters is not None:
            assert_keys_if_params_provided(parameters, [w_name, b_name])
            w = parameters[w_name]
            b = parameters[b_name]
        else:
            w = generate_init_weight(
                weight_shape=(
                    self.out_channels,
                    self.in_channels,
                    *self.kernel_size,
                ),
                initializer=self.param_init.initializer,
                mode=self.param_init.mode,
                stddev=self.param_init.weight_init_std,
            )
            b = generate_init_bias(
                bias_shape=(self.out_channels,),
                initializer="zeros",
            )

        return Conv2d(
            w=(w_name, w), b=(b_name, b), stride=self.stride, pad=self.pad
        )


@dataclass(frozen=True, kw_only=True)
class Conv2dGroupConfig(LayerConfig):
    """Configuration for the Conv2dGroup layer.

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
    """

    in_channels: int
    """The number of input channels."""

    out_channels: int
    """The number of output channels."""

    group: int

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    kernel_size: tuple[int, int]
    """The size of the kernel."""

    stride: int = 1
    """The stride of the kernel."""

    pad: int = 0
    """The padding size."""

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in", weight_init_std=None
    )

    def __post_init__(self) -> None:
        assert self.in_channels % self.group == 0
        assert self.out_channels % self.group == 0

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        w_names = [
            f"conv{self.param_suffix}_{idx}_w" for idx in range(self.group)
        ]
        b_names = [
            f"conv{self.param_suffix}_{idx}_b" for idx in range(self.group)
        ]
        if parameters is not None:
            assert_keys_if_params_provided(parameters, w_names + b_names)
            ws = [parameters[w_name] for w_name in w_names]
            bs = [parameters[b_name] for b_name in b_names]
        else:
            ws = [
                generate_init_weight(
                    weight_shape=(
                        int(self.out_channels / self.group),
                        int(self.in_channels / self.group),
                        *self.kernel_size,
                    ),
                    initializer=self.param_init.initializer,
                    mode=self.param_init.mode,
                    stddev=self.param_init.weight_init_std,
                )
                for _ in range(self.group)
            ]
            bs = [
                generate_init_bias(
                    bias_shape=(int(self.out_channels / self.group),),
                    initializer="zeros",
                )
                for _ in range(self.group)
            ]

        return Conv2dGroup(
            conv_layers=[
                Conv2d(
                    w=(w_names[idx], ws[idx]),
                    b=(b_names[idx], bs[idx]),
                    stride=self.stride,
                    pad=self.pad,
                )
                for idx in range(self.group)
            ]
        )


@dataclass(frozen=True, kw_only=True)
class DropoutConfig(LayerConfig):
    """Configuration for the Dropout layer."""

    dropout_ratio: float = 0.5
    """The ratio of the neurons to drop during training."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return Dropout(dropout_ratio=self.dropout_ratio)


@dataclass(frozen=True, kw_only=True)
class Dropout2dConfig(LayerConfig):
    """Configuration for the Dropout layer."""

    dropout_ratio: float
    """The ratio of the neurons to drop during training."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return Dropout2d(dropout_ratio=self.dropout_ratio)


@dataclass(frozen=True, kw_only=True)
class FlattenConfig(LayerConfig):
    """Configuration for the Flatten layer."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return Flatten()


@dataclass(frozen=True, kw_only=True)
class GlobalAvgPoolingConfig(LayerConfig):
    """Configuration for the GlobalAvgPooling layer."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return GlobalAvgPooling()


@dataclass(frozen=True, kw_only=True)
class MaxPool2dConfig(AvgPool2dConfig):
    """Configuration for the 2D max pooling layer."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return MaxPool2d(
            kenel_size=self.kernel_size,
            stride=self.stride,
            pad=self.pad,
        )


@dataclass(frozen=True, kw_only=True)
class ReLUConfig(LayerConfig):
    """Configuration for the ReLU layer."""

    inplace: bool = False

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return ReLU(inplace=self.inplace)


@dataclass(frozen=True, kw_only=True)
class ResBlockConfig(LayerConfig):
    """Configuration for the ResBlock layer.

    Diagram:
        input -> Conv1 (3x3) -> BatchNorm -> ReLU ->
                Conv2 (3x3) -> BatchNorm -> + -> ReLU -> output
          |_________________________________|
    """

    in_channel: int

    out_channel: int

    stride: int

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in", weight_init_std=None
    )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        conv1 = Conv2dConfig(
            in_channels=self.in_channel,
            out_channels=self.out_channel,
            param_suffix=f"{self.param_suffix}_1",
            kernel_size=(3, 3),
            stride=self.stride,
            pad=1,
            param_init=self.param_init,
        )
        batch_norm1 = BatchNorm2dConfig(
            num_feature=self.out_channel, param_suffix=f"{self.param_suffix}_1"
        )
        conv2 = Conv2dConfig(
            in_channels=self.out_channel,
            out_channels=self.out_channel,
            param_suffix=f"{self.param_suffix}_2",
            kernel_size=(3, 3),
            stride=1,
            pad=1,
            param_init=self.param_init,
        )
        batch_norm2 = BatchNorm2dConfig(
            num_feature=self.out_channel, param_suffix=f"{self.param_suffix}_2"
        )
        # Shortcut (skip connection) to match the dimensions of the output
        shortcut = []
        if self.stride > 1 or (self.in_channel != self.out_channel):
            shortcut_configs = [
                Conv2dConfig(
                    in_channels=self.in_channel,
                    out_channels=self.out_channel,
                    param_suffix=f"{self.param_suffix}_shortcut",
                    kernel_size=(1, 1),
                    stride=self.stride,
                    pad=0,
                    param_init=self.param_init,
                ),
                BatchNorm2dConfig(
                    num_feature=self.out_channel,
                    param_suffix=f"{self.param_suffix}_shortcut",
                ),
            ]
            shortcut = [
                config.create(parameters) for config in shortcut_configs
            ]
        return ResBlock(
            conv1=conv1.create(parameters),
            batch_norm1=batch_norm1.create(parameters),
            conv2=conv2.create(parameters),
            batch_norm2=batch_norm2.create(parameters),
            shortcut=shortcut,
        )


@dataclass(frozen=True, kw_only=True)
class ResBlocksConfig(LayerConfig):
    """Configuration for the ResBlock with multi layers.

    Diagram:
        input -> Conv1 (3x3) -> BatchNorm -> ReLU ->
                Conv2 (3x3) -> BatchNorm -> + -> ReLU -> output
          |_________________________________|
    """

    in_channel: int

    out_channel: int

    stride: int

    layer: int

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in", weight_init_std=None
    )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        layers = tuple(
            ResBlockConfig(
                in_channel=self.in_channel if idx == 0 else self.out_channel,
                out_channel=self.out_channel,
                stride=self.stride if idx == 0 else 1,
                param_suffix=f"{self.param_suffix}_{idx + 1}",
                param_init=self.param_init,
            ).create(parameters)
            for idx in range(self.layer)
        )
        return Sequential(layers=layers)


@dataclass(frozen=True, kw_only=True)
class BottleneckBlockConfig(LayerConfig):
    """Configuration for the BottleneckBlock layer.

    Diagram:
        input -> Conv1 (1x1, reduce channels) -> BatchNorm -> ReLU ->
                Conv2 (3x3, main conv) -> BatchNorm -> ReLU ->
                Conv3 (1x1, restore channels) -> BatchNorm -> + -> ReLU -> output
          |___________________________________________________|
    """

    in_channel: int

    bottle_channel: int

    out_channel: int

    stride: int

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in", weight_init_std=None
    )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        conv1 = Conv2dConfig(
            in_channels=self.in_channel,
            out_channels=self.bottle_channel,
            param_suffix=f"{self.param_suffix}_1",
            kernel_size=(1, 1),
            stride=1,
            pad=0,
            param_init=self.param_init,
        )
        batch_norm1 = BatchNorm2dConfig(
            num_feature=self.bottle_channel,
            param_suffix=f"{self.param_suffix}_1",
        )
        conv2 = Conv2dConfig(
            in_channels=self.bottle_channel,
            out_channels=self.bottle_channel,
            param_suffix=f"{self.param_suffix}_2",
            kernel_size=(3, 3),
            stride=self.stride,
            pad=1,
            param_init=self.param_init,
        )
        batch_norm2 = BatchNorm2dConfig(
            num_feature=self.bottle_channel,
            param_suffix=f"{self.param_suffix}_2",
        )
        conv3 = Conv2dConfig(
            in_channels=self.bottle_channel,
            out_channels=self.out_channel,
            param_suffix=f"{self.param_suffix}_3",
            kernel_size=(1, 1),
            stride=1,
            pad=0,
            param_init=self.param_init,
        )
        batch_norm3 = BatchNorm2dConfig(
            num_feature=self.out_channel, param_suffix=f"{self.param_suffix}_3"
        )
        shortcut = []
        if self.stride > 1 or (self.in_channel != self.out_channel):
            configs = [
                Conv2dConfig(
                    in_channels=self.in_channel,
                    out_channels=self.out_channel,
                    param_suffix=f"{self.param_suffix}_shortcut",
                    kernel_size=(1, 1),
                    stride=self.stride,
                    pad=0,
                    param_init=self.param_init,
                ),
                BatchNorm2dConfig(
                    num_feature=self.out_channel,
                    param_suffix=f"{self.param_suffix}_shortcut",
                ),
            ]
            shortcut = [config.create(parameters) for config in configs]
        return BottleneckBlock(
            conv1=conv1.create(parameters),
            conv2=conv2.create(parameters),
            conv3=conv3.create(parameters),
            batch_norm1=batch_norm1.create(parameters),
            batch_norm2=batch_norm2.create(parameters),
            batch_norm3=batch_norm3.create(parameters),
            shortcut=shortcut,
        )


@dataclass(frozen=True, kw_only=True)
class BottleneckBlocksConfig(LayerConfig):
    """Configuration for the BottleneckBlocks with multi layer.

    Diagram:
        input -> Conv1 (1x1, reduce channels) -> BatchNorm -> ReLU ->
                Conv2 (3x3, main conv) -> BatchNorm -> ReLU ->
                Conv3 (1x1, restore channels) -> BatchNorm -> + -> ReLU -> output
          |___________________________________________________|
    """

    in_channel: int

    bottle_channel: int

    out_channel: int

    stride: int

    layer: int

    param_suffix: str
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    param_init: ParameterInitConfig = ParameterInitConfig(
        initializer="he_normal", mode="fan_in", weight_init_std=None
    )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        layers = tuple(
            BottleneckBlockConfig(
                in_channel=self.in_channel if idx == 0 else self.out_channel,
                bottle_channel=self.bottle_channel,
                out_channel=self.out_channel,
                stride=self.stride if idx == 0 else 1,
                param_suffix=f"{self.param_suffix}_{idx + 1}",
                param_init=self.param_init,
            ).create(parameters)
            for idx in range(self.layer)
        )
        return Sequential(layers=layers)


@dataclass(frozen=True, kw_only=True)
class SequentialConfig(LayerConfig):
    """Configuration for the Sequential layer.

    diagram:
        input -- layers -- output
    """

    hidden_layer_configs: tuple[LayerConfig, ...]
    """The configurations of the hidden layers in order.

    If we provide the loss function with the last layer for the trainer
    independently, we do not need to include the last layer in this list.
    """

    load_params: str | None = None
    """The file to load the parameters for the network."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        params = decide_params(self.load_params, parameters)

        layers = create_layers(self.hidden_layer_configs, params)
        return Sequential(tuple(layers))


@dataclass(frozen=True, kw_only=True)
class SigmoidConfig(LayerConfig):
    """Configuration for the Sigmoid layer."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return Sigmoid()


@dataclass(frozen=True, kw_only=True)
class SoftmaxConfig(LayerConfig):
    """Configuration for the Softmax layer."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return Softmax()


@dataclass(frozen=True, kw_only=True)
class SoftmaxWithLossConfig(LayerConfig):
    """Configuration for the softmax with cross entropy loss."""

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        return SoftmaxWithLoss()


def create_layers(
    layer_configs: Sequence[LayerConfig],
    parameters: dict[str, NDArray[np.floating]] | None = None,
) -> list[Layer]:
    return [config.create(parameters) for config in layer_configs]


def assert_keys_if_params_provided(
    parameters: dict[str, NDArray[np.floating]],
    keys: Sequence[str],
) -> None:
    """Assert that the parameters are provided for the given keys."""
    missing_keys = [key for key in keys if (key not in parameters)]
    if missing_keys:
        raise ValueError(
            f"Parameters for {', '.join(missing_keys)} have to be provided."
        )


def decide_params(
    load_params: str | None,
    outside_params: dict[str, NDArray[np.floating]] | None,
) -> dict[str, NDArray[np.floating]] | None:
    if (load_params is not None) and (outside_params is not None):
        raise ValueError(
            "The parameters should not be provided when loading the parameters."
        )

    params = outside_params
    if load_params is not None:
        with open(load_params, "rb") as f:
            params = pickle.load(f)

    return params
