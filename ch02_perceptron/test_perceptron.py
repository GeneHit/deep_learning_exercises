import pytest

from ch02_perceptron.perceptron import AND, NAND, OR, XOR


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (0, 0, False),
        (0, 1, False),
        (1, 0, False),
        (1, 1, True),
    ],
)
def test_AND(x1: float, x2: float, expected: bool) -> None:
    assert AND(x1, x2) == expected


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (0, 0, True),
        (0, 1, True),
        (1, 0, True),
        (1, 1, False),
    ],
)
def test_NAND(x1: float, x2: float, expected: bool) -> None:
    assert NAND(x1, x2) == expected


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (0, 0, False),
        (0, 1, True),
        (1, 0, True),
        (1, 1, True),
    ],
)
def test_OR(x1: float, x2: float, expected: bool) -> None:
    assert OR(x1, x2) == expected


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (0, 0, False),
        (0, 1, True),
        (1, 0, True),
        (1, 1, False),
    ],
)
def test_XOR(x1: float, x2: float, expected: bool) -> None:
    assert XOR(x1, x2) == expected
