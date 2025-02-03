import numpy as np
import pytest

from ch06_learning_technique.b_weight_init import (
    generate_init_bias,
    generate_init_weight,
)
from common.default_type_array import (
    get_default_type,
    np_ones,
    np_zeros,
)


@pytest.mark.parametrize(
    "weight_shape, initializer, mode, stddev, expected_std_range",
    [
        (
            (64, 128),
            "he_normal",
            "fan_in",
            None,
            (np.sqrt(2.0 / 64) * 0.9, np.sqrt(2.0 / 64) * 1.1),
        ),
        (
            (64, 128),
            "he_uniform",
            "fan_out",
            None,
            (-np.sqrt(6.0 / 128), np.sqrt(6.0 / 128)),
        ),
        (
            (64, 128),
            "xavier_normal",
            "fan_in",
            None,
            (
                np.sqrt(1.0 / 64) * 0.9,
                np.sqrt(1.0 / 64) * 1.1,
            ),
        ),
        (
            (64, 128),
            "xavier_uniform",
            "fan_avg",
            None,
            (-np.sqrt(3.0 / ((64 + 128) / 2)), np.sqrt(3.0 / ((64 + 128) / 2))),
        ),
        (
            (2, 3, 4, 5),
            "he_normal",
            "fan_out",
            None,
            (
                np.sqrt(2.0 / (2 * 4 * 5)) * 0.9,
                np.sqrt(2.0 / (2 * 4 * 5)) * 1.1,
            ),
        ),
        (
            (2, 3, 64, 64),
            "he_uniform",
            "fan_in",
            None,
            (-np.sqrt(6.0 / (3 * 64 * 64)), np.sqrt(6.0 / (3 * 64 * 64))),
        ),
        (
            (2, 3, 32, 128),
            "xavier_normal",
            "fan_avg",
            None,
            (
                np.sqrt(1.0 / ((2 + 3) * 32 * 128 / 2.0)) * 0.9,
                np.sqrt(1.0 / ((2 + 3) * 32 * 128 / 2.0)) * 1.1,
            ),
        ),
        (
            (2, 5, 16, 32),
            "xavier_uniform",
            "fan_avg",
            None,
            (
                -np.sqrt(3.0 / ((2 + 5) * 16 * 32 / 2.0)),
                np.sqrt(3.0 / ((2 + 5) * 16 * 32 / 2.0)),
            ),
        ),
        (
            (2, 3, 8, 8),
            "he_normal",
            "fan_in",
            None,
            (
                np.sqrt(2.0 / (3 * 8 * 8)) * 0.9,
                np.sqrt(2.0 / (3 * 8 * 8)) * 1.1,
            ),
        ),
        ((64, 128), "normal", "fan_in", 0.01, (-0.02, 0.02)),
        (
            (64, 128),
            "uniform",
            "fan_in",
            0.01,
            (-0.01 * np.sqrt(3), 0.01 * np.sqrt(3)),
        ),
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


class TestGenerateInitBias:
    """Test suite for generate_init_bias function"""

    @pytest.mark.parametrize(
        "bias_shape, initializer, stddev",
        [
            ((2, 2), "zeros", None),  # Zeros initializer
            ((2,), "zeros", None),  # Zeros initializer
            ((3, 3), "ones", None),  # Ones initializer
            ((3,), "ones", None),  # Ones initializer
            ((32, 32), "normal", 0.5),  # Normal initializer
            ((3, 3), "uniform", 0.1),  # Uniform initializer
        ],
    )
    def test_valid_cases(
        self,
        bias_shape: tuple[int, ...],
        initializer: str,
        stddev: float | None,
    ) -> None:
        """Test valid cases for different initializers"""
        bias = generate_init_bias(bias_shape, initializer, stddev)
        assert bias.dtype == get_default_type()
        assert bias.shape == bias_shape

        if "ones" in initializer:
            np.testing.assert_array_equal(bias, np_ones(bias_shape))
        elif "zeros" in initializer:
            np.testing.assert_array_equal(bias, np_zeros(bias_shape))
        else:
            assert stddev is not None
            # Check the standard deviation range for normal distributions
            if "normal" in initializer:
                std = np.std(bias)
                assert 0.9 * stddev <= std <= 1.1 * stddev, (
                    f"Standard deviation {std} is not in expected range "
                    f"{0.9 * stddev} - {1.1 * stddev}"
                )

            # Check the range for uniform distributions
            if "uniform" in initializer:
                min_val, max_val = np.min(bias), np.max(bias)
                assert -stddev <= min_val, (
                    f"Min value {min_val} is less than expected {-stddev}"
                )
                assert max_val <= stddev, (
                    f"Max value {max_val} is greater than expected {stddev}"
                )

    @pytest.mark.parametrize(
        "initializer, stddev",
        [
            ("normal", None),  # Missing stddev for normal
            ("uniform", None),  # Missing stddev for uniform
            ("normal", -0.5),  # Invalid stddev (negative)
            ("uniform", 0),  # Invalid stddev (zero)
        ],
    )
    def test_invalid_stddev(
        self, initializer: str, stddev: float | None
    ) -> None:
        """Test cases where stddev is required but not valid"""
        with pytest.raises(
            ValueError, match="stddev must be provided for normal/uniform."
        ):
            generate_init_bias((2, 2), initializer, stddev)

    def test_invalid_initializer(self) -> None:
        """Test case where an invalid initializer is provided"""
        with pytest.raises(AssertionError):
            generate_init_bias((2, 2), "invalid_initializer")
