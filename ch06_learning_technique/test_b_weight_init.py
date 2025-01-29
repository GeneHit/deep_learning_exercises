import numpy as np
import pytest

from ch06_learning_technique.b_weight_init import generate_init_weight
from common.default_type_array import get_default_type


@pytest.mark.parametrize(
    "weight_shape, initializer, mode, stddev, expected_std_range",
    [
        (
            (64, 128),
            "he_normal",
            "fan_in",
            None,
            (np.sqrt(2 / 64) * 0.9, np.sqrt(2 / 64) * 1.1),
        ),
        (
            (64, 128),
            "he_uniform",
            "fan_in",
            None,
            (-np.sqrt(6 / 64), np.sqrt(6 / 64)),
        ),
        (
            (64, 128),
            "xavier_normal",
            "fan_avg",
            None,
            (
                np.sqrt(1 / ((64 + 128) / 2)) * 0.9,
                np.sqrt(1 / ((64 + 128) / 2)) * 1.1,
            ),
        ),
        (
            (64, 128),
            "xavier_uniform",
            "fan_avg",
            None,
            (-np.sqrt(3 / ((64 + 128) / 2))),
            np.sqrt(3 / ((64 + 128) / 2)),
        ),
        (
            (3, 3, 3, 3),
            "he_normal",
            "fan_out",
            None,
            (np.sqrt(2 / (3 * 3 * 3)) * 0.9, np.sqrt(2 / (3 * 3 * 3)) * 1.1),
        ),
        (
            (3, 3, 64, 64),
            "he_uniform",
            "fan_in",
            None,
            (-np.sqrt(6 / (3 * 3 * 64)), np.sqrt(6 / (3 * 3 * 64))),
        ),
        (
            (3, 3, 32, 128),
            "xavier_normal",
            "fan_avg",
            None,
            (
                np.sqrt(1 / (3 * 3 * (32 + 128) / 2)) * 0.9,
                np.sqrt(1 / (3 * 3 * (32 + 128) / 2)) * 1.1,
            ),
        ),
        (
            (5, 5, 16, 32),
            "xavier_uniform",
            "fan_avg",
            None,
            (-np.sqrt(3 / (5 * 5 * (16 + 32) / 2))),
            np.sqrt(3 / (5 * 5 * (16 + 32) / 2)),
        ),
        (
            (7, 7, 8, 8),
            "he_normal",
            "fan_in",
            None,
            (np.sqrt(2 / (7 * 7 * 8)) * 0.9, np.sqrt(2 / (7 * 7 * 8)) * 1.1),
        ),
        ((64, 128), "normal", "fan_in", 0.01, (-0.02, 0.02)),
        ((64, 128), "uniform", "fan_in", 0.01, (-0.01, 0.01)),
    ],
)
def test_generate_init_weight(
    weight_shape: tuple[int, ...],
    initializer: str,
    mode: str,
    stddev: float | None,
    expected_std_range: tuple[float, float],
) -> None:
    """Test the generate_init_weight function with different parameters."""
    weights = generate_init_weight(weight_shape, initializer, mode, stddev)

    # Ensure the shape of the weights matches the expected shape
    assert weights.shape == weight_shape, (
        f"Shape mismatch: {weights.shape} != {weight_shape}"
    )

    # Ensure the dtype of the weights is correct
    assert weights.dtype == get_default_type(), (
        f"Dtype mismatch: {weights.dtype} != {get_default_type()}"
    )

    # Check the standard deviation range for normal distributions
    if "normal" in initializer:
        std = np.std(weights)
        assert expected_std_range[0] <= std <= expected_std_range[1], (
            f"Standard deviation {std} is not in expected range {expected_std_range}"
        )

    # Check the range for uniform distributions
    if "uniform" in initializer:
        min_val, max_val = np.min(weights), np.max(weights)
        assert expected_std_range[0] <= min_val, (
            f"Min value {min_val} is less than expected {expected_std_range[0]}"
        )
        assert max_val <= expected_std_range[1], (
            f"Max value {max_val} is greater than expected {expected_std_range[1]}"
        )


@pytest.mark.parametrize(
    "weight_shape, initializer, mode, stddev, expected_error",
    [
        (
            (-64, 128),
            "he_normal",
            "fan_in",
            None,
            ValueError,
        ),  # Negative dimensions
        (
            (64, 128),
            "unknown_initializer",
            "fan_in",
            None,
            ValueError,
        ),  # Invalid initializer
        (
            (64, 128),
            "he_normal",
            "unknown_mode",
            None,
            ValueError,
        ),  # Invalid mode
        ((64, 128), "normal", "fan_in", -0.01, ValueError),  # Negative stddev
    ],
)
def test_generate_init_weight_invalid(
    weight_shape: tuple[int, ...],
    initializer: str,
    mode: str,
    stddev: float | None,
    expected_error: type,
) -> None:
    """Test the generate_init_weight function with invalid parameters."""
    with pytest.raises(expected_error):
        generate_init_weight(weight_shape, initializer, mode, stddev)
