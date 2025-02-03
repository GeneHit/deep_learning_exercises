import numpy as np
from numpy.typing import NDArray


class AddLayer:
    """Layer that adds two inputs.

    The grapfical representation of the layer is:

        x  ----> (+) ----> x + y
        y  ----/
    """

    def forward(
        self, x: NDArray[np.floating], y: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        return x + y

    def backward(
        self, dout: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward pass of the layer."""
        return dout, dout


class MulLayer:
    """Layer that multiplies two inputs.

    The graphical representation of the layer is:

        x  ----> (*) ----> x * y
        y  ----/
    """

    def __init__(self) -> None:
        self._x: NDArray[np.floating] | None = None
        self._y: NDArray[np.floating] | None = None

    def forward(
        self, x: NDArray[np.floating], y: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        self._x = x
        self._y = y
        return x * y

    def backward(
        self, dout: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward pass of the layer."""
        assert self._x is not None and self._y is not None
        dx = dout * self._y.T
        dy = self._x.T * dout
        return dx, dy


class DivisorLayer:
    """Layer that divides two inputs.

    The graphical representation of the layer is:

        x  ----> (/) ----> x / y
        y  ----/
    """

    def __init__(self) -> None:
        self._x: NDArray[np.floating] | None = None
        self._y: NDArray[np.floating] | None = None

    def forward(
        self, x: NDArray[np.floating], y: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        self._x = x
        self._y = y
        return x / y

    def backward(
        self, dout: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward pass of the layer."""
        assert self._x is not None and self._y is not None
        dx = dout / self._y
        dy = -self._x * (dout / self._y**2)
        return dx, dy


class ExpoLayer:
    """Layer that raises the Euler's number to the power of the input.

    The graphical representation of the layer is:

        x  ----> (exp) ----> exp(x)
    """

    def __init__(self) -> None:
        self._exp_x: NDArray[np.floating] | None = None

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        self._exp_x = np.exp(x)
        assert self._exp_x is not None  # for mypy
        return self._exp_x

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        assert self._exp_x is not None
        return self._exp_x * dout
