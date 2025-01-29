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
from common.base import Layer, LayerConfig


@dataclass(frozen=True, kw_only=True)
class AffineConfig(LayerConfig):
    """Configuration for the Affine layer."""

    in_size: int
    """The input size of the layer."""

    out_size: int
    """The output size of the layer."""

    initializer: str = "he_normal"
    """The initializer for the weights of the layer.

    Options:
        - "he_normal": He normal initializer.
        - "he_uniform": He uniform initializer.
        - "xavier_normal": Xavier normal initializer.
        - "xavier_uniform": Xavier uniform initializer.
        - "normal": Normal distribution with weight_init_std.
        - "uniform": Uniform distribution with weight_init_std.
    """

    weight_init_std: float | None = None
    """The standard deviation for the distribution of the weights.

    It is only used when the initializer is "normal" or "uniform".
    """

    param_suffix: str = ""
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    def __post_init__(self) -> None:
        # checker
        if self.initializer in {"normal", "uniform"}:
            assert self.weight_init_std is not None, (
                "The weight_init_std have to be provided for the initializer."
            )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        w_name = f"W_{self.param_suffix}"
        b_name = f"b_{self.param_suffix}"
        if parameters is not None:
            assert_keys_if_params_provided(parameters, [w_name, b_name])
            w = parameters[w_name]
            b = parameters[b_name]
        else:
            w = generate_init_weight(
                weight_shape=(self.in_size, self.out_size),
                initializer=self.initializer,
                stddev=self.weight_init_std,
            )
            b = generate_init_bias(
                bias_shape=(self.out_size,), initializer="zeros"
            )

        return Affine(W=(w_name, w), b=(b_name, b))


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

    momentum: float = 0.9
    """The momentum for the moving average."""

    affine: bool = True
    """If True, learnable affine parameters (gamma and beta) are used."""

    track_running_stats: bool = True
    """If True, running mean and variance are tracked during training."""

    eps: float = 1e-5
    """A small constant added to the denominator for numerical stability."""

    param_suffix: str = ""
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `gamma_{param_suffix}` and `beta_{param_suffix}`.
    """

    def _get_params(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> tuple[
        tuple[str, NDArray[np.floating]], tuple[str, NDArray[np.floating]]
    ]:
        gamma_name = f"gamma_{self.param_suffix}"
        beta_name = f"beta_{self.param_suffix}"
        if parameters is not None:
            assert_keys_if_params_provided(parameters, [gamma_name, beta_name])
            gamma = parameters[gamma_name]
            beta = parameters[beta_name]
        else:
            gamma = generate_init_bias(
                bias_shape=(self.num_feature,), initializer="ones"
            )
            beta = generate_init_bias(
                bias_shape=(self.num_feature,), initializer="zeros"
            )

        return (gamma_name, gamma), (beta_name, beta)

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        gamma, beta = self._get_params(parameters)
        return BatchNorm1d(
            gamma=gamma,
            beta=beta,
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
        gamma, beta = self._get_params(parameters)
        return BatchNorm2d(
            gamma=gamma,
            beta=beta,
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

    kernel_size: tuple[int, int]
    """The size of the kernel."""

    stride: int = 1
    """The stride of the kernel."""

    pad: int = 0
    """The padding size."""

    initializer: str = "he_normal"
    """The initializer for the weights of the layer.

    Options:
        - "he_normal": He normal initializer.
        - "he_uniform": He uniform initializer.
        - "xavier_normal": Xavier normal initializer.
        - "xavier_uniform": Xavier uniform initializer.
        - "normal": Normal distribution with weight_init_std.
        - "uniform": Uniform distribution with weight_init_std.
    """

    weight_init_std: float | None = None
    """The standard deviation for the distribution of the weights.

    It is only used when the initializer is "normal" or "uniform".
    """

    param_suffix: str = ""
    """The suffix for the layer's parameter, which should be unique.

    The name of the parameter will be `W_{param_suffix}` and `b_{param_suffix}`.
    """

    def __post_init__(self) -> None:
        # checker
        if self.initializer in {"normal", "uniform"}:
            assert self.weight_init_std is not None, (
                "The weight_init_std have to be provided for the initializer."
            )

    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        w_name = f"W_{self.param_suffix}"
        b_name = f"b_{self.param_suffix}"
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
                initializer=self.initializer,
                stddev=self.weight_init_std,
            )
            b = generate_init_bias(
                bias_shape=(self.out_channels,),
                initializer="zeros",
            )

        return Conv2d(
            W=(w_name, w), b=(b_name, b), stride=self.stride, pad=self.pad
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
