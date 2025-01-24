import numpy as np
import pytest

from ch06_learning_technique.b_weight_init import generate_init_weight


@pytest.mark.parametrize(
    "layer_type, weight_shape, activation, distribution, stddev, expected_fan_in, expected_variance",
    [
        # Case 1: Conv layer with ReLU (He initialization, normal distribution)
        (
            "conv",
            (16, 3, 3, 3),
            "relu",
            "normal",
            None,
            3 * 3 * 3,
            2 / (3 * 3 * 3),
        ),
        # Case 2: Affine layer with Sigmoid (Xavier initialization, normal distribution)
        ("affine", (256, 128), "sigmoid", "normal", None, 256, 1 / 256),
        # Case 3: Conv layer with Tanh (Xavier initialization, uniform distribution)
        (
            "conv",
            (32, 3, 5, 5),
            "tanh",
            "uniform",
            None,
            3 * 5 * 5,
            3 / (3 * 5 * 5),
        ),
        # Case 4: Affine layer with custom stddev
        (
            "affine",
            (64, 32),
            "relu",
            "normal",
            0.1,
            64,
            0.01,
        ),  # stddev=0.1 -> variance=0.01
    ],
)
def test_generate_init_weight(
    layer_type: str,
    weight_shape: tuple[int, ...],
    activation: str,
    distribution: str,
    stddev: float | None,
    expected_fan_in: int,
    expected_variance: float,
) -> None:
    """Test generate_init_weight with different layer types, activations, and distributions."""
    weights = generate_init_weight(
        layer_type=layer_type,
        weight_shape=weight_shape,
        activation=activation,
        distribution=distribution,
        stddev=stddev,
    )

    # Check shape
    assert weights.shape == weight_shape, (
        f"Expected shape {weight_shape}, got {weights.shape}"
    )

    # Check variance
    if stddev is not None:
        expected_variance = stddev**2

    actual_variance = np.var(weights)
    np.testing.assert_almost_equal(
        actual_variance,
        expected_variance,
        decimal=2,
        err_msg=f"Expected variance ~{expected_variance}, got {actual_variance}",
    )

    # Check distribution type
    if distribution == "uniform":
        assert np.all(
            (weights >= -np.sqrt(3 * expected_variance))
            & (weights <= np.sqrt(3 * expected_variance))
        ), (
            f"Weights out of expected uniform range for variance {expected_variance}"
        )
    elif distribution == "normal":
        # Normal distribution tests are approximate
        assert abs(np.mean(weights)) < 0.1, (
            f"Expected mean ~0, got {np.mean(weights)}"
        )


@pytest.mark.parametrize(
    "invalid_layer_type, weight_shape, activation, distribution",
    [
        ("invalid", (16, 3, 3, 3), "relu", "normal"),  # Invalid layer type
        ("conv", (16, 3, 3), "relu", "normal"),  # Invalid weight shape for conv
        ("affine", (256,), "relu", "normal"),  # Invalid weight shape for affine
        ("affine", (256, 128), "invalid", "normal"),  # Invalid activation
        ("affine", (256, 128), "relu", "invalid"),  # Invalid distribution
    ],
)
def test_generate_init_weight_invalid_inputs(
    invalid_layer_type: str,
    weight_shape: tuple[int, ...],
    activation: str,
    distribution: str,
) -> None:
    """Test generate_init_weight with invalid inputs."""
    with pytest.raises(ValueError):
        generate_init_weight(
            layer_type=invalid_layer_type,
            weight_shape=weight_shape,
            activation=activation,
            distribution=distribution,
        )
