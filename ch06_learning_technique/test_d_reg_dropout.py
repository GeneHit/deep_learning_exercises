import numpy as np
import pytest

from ch06_learning_technique.d_reg_dropout import Dropout, Dropout2d
from common.default_type_array import np_randn


@pytest.mark.parametrize(
    "input_shape,dropout_ratio,training",
    [
        ((2, 3), 0.5, True),
        ((3, 4), 0.3, True),
        ((2, 3), 0.5, False),
        ((4, 5), 0.0, True),
    ],
)
def test_dropout(
    input_shape: tuple[int, ...], dropout_ratio: float, training: bool
) -> None:
    # Setup
    dropout = Dropout(dropout_ratio=dropout_ratio)
    x = np_randn(input_shape)

    # Forward pass
    dropout.train(flag=training)
    output = dropout.forward(x)
    assert output.shape == input_shape

    if not training:
        # In eval mode, output should equal input
        np.testing.assert_array_almost_equal(output, x)
        return None

    if dropout_ratio == 0.0:
        # No dropout
        np.testing.assert_array_almost_equal(output, x)
    elif dropout_ratio == 1.0:
        # Full dropout
        np.testing.assert_array_almost_equal(output, np.zeros_like(x))
    else:
        # Check if values are either 0 or scaled up by 1/(1-p)
        mask = output != 0
        if mask.any():
            scaled_x = x[mask] / (1 - dropout_ratio)
            np.testing.assert_array_almost_equal(output[mask], scaled_x)

    # Backward pass
    dout = np_randn(input_shape)
    dx = dropout.backward(dout)
    assert dx.shape == input_shape

    # Dropout has no parameters
    assert len(dropout.named_params()) == 0
    # Dropout has no gradients
    assert len(dropout.param_grads()) == 0


class TestDropout2d:
    @pytest.mark.parametrize(
        "input_shape, dropout_ratio, inplace",
        [
            ((2, 3, 32, 32), 0.5, False),
            ((1, 4, 16, 16), 0.3, True),
            ((4, 8, 8, 8), 0.7, False),
        ],
    )
    def test_dropout2d_forward_train(
        self, input_shape: tuple[int, ...], dropout_ratio: float, inplace: bool
    ) -> None:
        dropout = Dropout2d(dropout_ratio=dropout_ratio, inplace=inplace)
        dropout.train(True)  # Set to training mode
        x = np_randn(input_shape)
        x_orig = x.copy()

        # Forward pass
        output = dropout.forward(x)

        # Check output shape
        assert output.shape == input_shape

        # Check if inplace operation worked as expected
        if inplace:
            assert np.may_share_memory(x, output)
        else:
            assert not np.may_share_memory(x, output)
            assert np.array_equal(x, x_orig)

    @pytest.mark.parametrize(
        "input_shape, dropout_ratio",
        [
            ((2, 3, 32, 32), 0.5),
            ((1, 4, 16, 16), 0.3),
        ],
    )
    def test_dropout2d_forward_eval(
        self, input_shape: tuple[int, ...], dropout_ratio: float
    ) -> None:
        dropout = Dropout2d(dropout_ratio=dropout_ratio)
        dropout.train(False)  # Set to evaluation mode
        x = np_randn(input_shape)

        # Forward pass in eval mode should return input unchanged
        output = dropout.forward(x)
        assert np.array_equal(output, x)

    @pytest.mark.parametrize(
        "input_shape, dropout_ratio",
        [
            ((2, 3, 32, 32), 0.5),
            ((1, 4, 16, 16), 0.3),
        ],
    )
    def test_dropout2d_backward(
        self, input_shape: tuple[int, ...], dropout_ratio: float
    ) -> None:
        dropout = Dropout2d(dropout_ratio=dropout_ratio)
        dropout.train(True)

        x = np_randn(input_shape)
        # Forward pass to set up the mask
        dropout.forward(x)

        # Backward pass
        dout = np_randn(input_shape)
        dx = dropout.backward(dout)

        # Check gradient shape
        assert dx.shape == input_shape

        # Check parameter gradients
        grads = dropout.param_grads()
        assert isinstance(grads, dict)
        assert len(grads) == 0  # Dropout has no parameters

    def test_dropout2d_invalid_backward(self) -> None:
        dropout = Dropout2d()
        with pytest.raises(AssertionError):
            # Should raise error if backward called before forward
            dropout.backward(np_randn((2, 3, 32, 32)))
