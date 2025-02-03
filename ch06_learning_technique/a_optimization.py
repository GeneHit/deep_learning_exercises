import numpy as np
from numpy.typing import NDArray

from common.base import Optimizer
from common.default_type_array import np_float, np_zeros_like

EPSILON = np_float(1e-8)


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    The formulation is: params = params - lr * grads
    """

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the SGD optimizer.

        Parameters:
            lr (float): Learning rate.
        """
        self._lr = np_float(lr)

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        for key in grads.keys():
            params[key] -= self._lr * grads[key]


class Momentum(Optimizer):
    """Momentum optimizer.

    The formulation is as follows:

        m_t = beta * m_{t-1} - self._lr * grads
        params = params + m_t

    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        """Initialize the Momentum optimizer.

        Parameters:
            lr (float): Learning rate.
            beta (float): Momentum factor.
        """
        self._lr = np_float(lr)
        self._beta = np_float(beta)
        self._m: dict[str, NDArray[np.floating]] | None = None

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        if self._m is None:
            self._m = {key: np_zeros_like(val) for key, val in params.items()}

        for key in grads.keys():
            self._m[key] = self._beta * self._m[key] - self._lr * grads[key]
            params[key] += self._m[key]


class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    The formulation is as follows:

        v += grads * grads
        params = params - lr * grads / (np.sqrt(v) + 1e-8)

    """

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the AdaGrad optimizer.

        Parameters:
            lr (float): Learning rate.
        """
        self._lr = np_float(lr)
        self._v: dict[str, NDArray[np.floating]] | None = None

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        if self._v is None:
            self._v = {key: np_zeros_like(val) for key, val in params.items()}

        for key in grads.keys():
            self._v[key] += grads[key] * grads[key]
            params[key] -= (
                self._lr * grads[key] / (np.sqrt(self._v[key]) + EPSILON)
            )


class RMSProp(Optimizer):
    """RMSProp optimizer.

    The formulation is as follows:

        v_t = decay_rate * v_{t-1} + (1 - decay_rate) * grads * grads
        params = params - lr * grads / (np.sqrt(v_t) + 1e-8)

    """

    def __init__(self, lr: float = 0.01, decay_rate: float = 0.99) -> None:
        """Initialize the RMSProp optimizer.

        Parameters:
            lr (float): Learning rate.
            decay_rate (float): Decay rate.
        """
        self._lr = np_float(lr)
        self._decay_rate = np_float(decay_rate)
        self._v: dict[str, NDArray[np.floating]] | None = None

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        if self._v is None:
            self._v = {key: np_zeros_like(val) for key, val in params.items()}

        for key in grads.keys():
            self._v[key] = (
                self._decay_rate * self._v[key]
                + (1 - self._decay_rate) * grads[key] * grads[key]
            )
            params[key] -= (
                self._lr * grads[key] / (np.sqrt(self._v[key]) + EPSILON)
            )


class Adam(Optimizer):
    """Adam optimizer.

    The formulation is as follows:

        m_t = beta1 * m_{t-1} + (1 - beta1) * grads
        v_t = beta2 * v_{t-1} + (1 - beta2) * grads * grads
        m_hat = m_t / (1 - beta1)
        v_hat = v_t / (1 - beta2)
        params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

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
        self._lr = np_float(lr)
        self._beta1 = np_float(beta1)
        self._beta2 = np_float(beta2)
        self._m: dict[str, NDArray[np.floating]] | None = None
        self._v: dict[str, NDArray[np.floating]] | None = None

    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """See the base class."""
        if self._m is None:
            self._m = {key: np_zeros_like(val) for key, val in params.items()}
        if self._v is None:
            self._v = {key: np_zeros_like(val) for key, val in params.items()}

        for key in grads.keys():
            self._m[key] = (
                self._beta1 * self._m[key]
                + (np_float(1) - self._beta1) * grads[key]
            )
            self._v[key] = (
                self._beta2 * self._v[key]
                + (np_float(1) - self._beta2) * grads[key] * grads[key]
            )
            # In the initial stage, it tends to be biased toward 0, so bias
            # correction is needed.
            m_hat = self._m[key] / (np_float(1) - self._beta1)
            v_hat = self._v[key] / (np_float(1) - self._beta2)
            params[key] -= self._lr * m_hat / (np.sqrt(v_hat) + EPSILON)
