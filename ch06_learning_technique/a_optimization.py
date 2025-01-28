import numpy as np
from numpy.typing import NDArray

from common.base import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    The formulation is: params = params - lr * grads
    """

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the SGD optimizer.

        Parameters:
            lr (float): Learning rate.
        """
        self._lr = lr

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        raise NotImplementedError


class Momentum(Optimizer):
    """Momentum optimizer.

    The formulation is as follows:

        v = momentum * v - lr * grads
        params = params + v

    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        """Initialize the Momentum optimizer.

        Parameters:
            lr (float): Learning rate.
            momentum (float): Momentum factor.
        """
        self._lr = lr
        self._momentum = momentum

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        raise NotImplementedError


class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    The formulation is as follows:

        h += grads * grads
        params = params - lr * grads / (np.sqrt(h) + 1e-7)

    """

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the AdaGrad optimizer.

        Parameters:
            lr (float): Learning rate.
        """
        self._lr = lr

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        raise NotImplementedError


class RMSProp(Optimizer):
    """RMSProp optimizer.

    The formulation is as follows:

        h = decay_rate * h + (1 - decay_rate) * grads * grads
        params = params - lr * grads / (np.sqrt(h) + 1e-7)

    """

    def __init__(self, lr: float = 0.01, decay_rate: float = 0.99) -> None:
        """Initialize the RMSProp optimizer.

        Parameters:
            lr (float): Learning rate.
            decay_rate (float): Decay rate.
        """
        self._lr = lr
        self._decay_rate = decay_rate

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        raise NotImplementedError


class Adam(Optimizer):
    """Adam optimizer.

    The formulation is as follows:

        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads * grads
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-7)

    """

    def __init__(
        self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        """Initialize the Adam optimizer.

        Parameters:
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for the first moment.
            beta2 (float): Exponential decay rate for the second moment.
        """
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        raise NotImplementedError
