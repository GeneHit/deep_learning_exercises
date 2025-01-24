import pytest

from ch06_learning_technique.b_weight_init import generate_init_weight

ATOL = 1e-2


@pytest.mark.parametrize(
    "shape, method, stddev",
    [
        ((3, 2), "xavier", None),
        ((3, 2), "he", None),
        ((5, 4), "xavier", None),
        ((5, 4), "he", None),
        ((3, 2), "", 0.1),
        ((5, 4), "", 0.2),
        ((3, 2), "he", 0.1),
        ((5, 4), "xavier", 0.2),
        ((3, 5, 4), "he", None),
    ],
)
def test_generate_init_weight(
    shape: tuple[int, ...],
    method: str,
    stddev: float | None,
) -> None:
    weights = generate_init_weight(shape, method, stddev)

    # TODO: it is not enough to check the shape of the weights
    assert weights.shape == shape
