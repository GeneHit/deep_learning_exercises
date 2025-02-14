import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from common.default_type_array import np_randn
from common.layer_config import BatchNorm2dConfig


class TestBatchNorm2d:
    @pytest.mark.parametrize(
        "batch_size, num_channels, height, width, momentum",
        [
            (4, 16, 32, 32, 0.1),  # Standard case
            (2, 8, 28, 28, 0.1),  # Smaller batch size
            (8, 32, 16, 16, 0.9),  # Larger number of channels
        ],
    )
    def test_batch_norm2d_forward(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        momentum: float,
    ) -> None:
        """Test that the forward pass of BatchNorm2d correctly normalizes the input."""
        # Create BatchNorm2d layer
        batch_norm = BatchNorm2dConfig(
            num_feature=num_channels,
            param_suffix="1",
            momentum=momentum,
        ).create()

        # Generate random input tensor
        x = np_randn(shape=(batch_size, num_channels, height, width))

        # Forward pass
        output = batch_norm.forward(x)

        # Check output shape
        assert output.shape == x.shape

        # Check that the mean is close to zero and variance is close to one (approximately)
        batch_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
        batch_var = np.var(output, axis=(0, 2, 3), keepdims=True)

        assert_almost_equal(batch_mean, 0, decimal=1)
        assert_almost_equal(batch_var, 1, decimal=1)

    @pytest.mark.parametrize(
        "batch_size, num_channels, height, width",
        [
            (4, 16, 32, 32),  # Standard case
            (2, 8, 28, 28),  # Smaller batch size
            (8, 32, 16, 16),  # Larger number of channels
        ],
    )
    def test_batch_norm2d_backward(
        self, batch_size: int, num_channels: int, height: int, width: int
    ) -> None:
        """Test that the backward pass of BatchNorm2d computes gradients correctly."""
        batch_norm = BatchNorm2dConfig(
            num_feature=num_channels, param_suffix="1"
        ).create()

        # Generate random input tensor
        x = np_randn(shape=(batch_size, num_channels, height, width))
        # Forward pass
        output = batch_norm.forward(x)

        # Generate random dout with the same shape as the output
        dout = np_randn(output.shape)
        # Backward pass
        dx = batch_norm.backward(dout)
        # Check that the gradient shape matches the input shape
        assert dx.shape == x.shape

        # Get gradients
        grads = batch_norm.param_grads()
        # Check that gradients for each parameter exist
        assert isinstance(grads, dict)
        assert len(grads) > 0  # There should be some parameters with gradients
        for value in grads.values():
            assert value.shape == (1, num_channels, 1, 1)

        for value in batch_norm.named_params().values():
            assert value.shape == (1, num_channels, 1, 1)
