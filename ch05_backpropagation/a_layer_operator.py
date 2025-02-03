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
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(
        self, dout: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")


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
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(
        self, dout: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")


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
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(
        self, dout: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")


class ExpoLayer:
    """Layer that raises the Euler's number to the power of the input.

    The graphical representation of the layer is:

        x  ----> (exp) ----> exp(x)
    """

    def __init__(self) -> None:
        self._exp_x: NDArray[np.floating] | None = None

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer."""
        raise NotImplementedError("The forward method is not implemented yet.")

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer."""
        raise NotImplementedError("The backward method is not implemented yet.")
