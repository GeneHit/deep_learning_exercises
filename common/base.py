import abc
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class Layer(abc.ABC):
    """Base class for neural network layers."""

    @abc.abstractmethod
    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters by the outside
        inplace operation +=, -=.
        """

    def train(self, flag: bool) -> None:
        """Set the training flag of the layer.

        During training, some layer may need to change their behavior, for
        example, dropout layer.

        Parameters:
            flag : bool
                Training flag. During training, the flag cann be set to True
                if neccesary.
        """
        pass

    def forward_to_loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating | np.integer],
    ) -> float:
        """Forward pass of the layer.

        BE CAREFUL: This method is used for the layers that are used in the
        loss layer, like the softmax layer.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating | np.integer]): Target output.

        Returns:
            float: Loss value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass of the layer.

        Parameters:
            x (NDArray[np.floating]): Input data.

        Returns:
            NDArray[np.floating]: Output data.
        """

    @abc.abstractmethod
    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """Backward pass of the layer.

        Parameters:
            dout (NDArray[np.floating]): Gradient of the loss.

        Returns:
            NDArray[np.floating]: Gradient of the input, given to next layer.
        """

    @abc.abstractmethod
    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters.

        This have to be called after the backward pass.
        """


@dataclass(frozen=True, kw_only=True)
class LayerConfig(abc.ABC):
    """Base class for the configuration of the layers."""

    @abc.abstractmethod
    def create(
        self, parameters: dict[str, NDArray[np.floating]] | None = None
    ) -> Layer:
        """Create the layer based on the configuration.

        Every special difinited layer should have a corresponding config.

        Parameters:
            parameters : dict[str, NDArray[np.floating]]
                Dictionary of parameters for the layer. If provided, it will
                be used for the initialization of the layer, instead of using
                the provided configured initializer. This is useful for loading
                the trained parameters.
        """


class Optimizer(abc.ABC):
    """Base class for all optimizers."""

    @abc.abstractmethod
    def one_step(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """Update the parameters using the gradients once.

        1. Have to use the inplace operation to update the parameters.
        2. Use the key of grads to update the corresponding parameters,
            because the params may have other parameters that aren't necessary
            to be updated, like the moving average of the batch normalization.

        Parameters:
            params : dict[str, NDArray[np.floating]]
                Dictionary of parameters to be updated.
            grads : dict[str, NDArray[np.floating]]
                Dictionary of gradients for the parameters.
        """


class Trainer(abc.ABC):
    """A base for the trainer of the neural network."""

    @abc.abstractmethod
    def train(self) -> None:
        """Train the network.

        Parameters:
            evaluate : bool
                Whether evaluate the network by train/test data every epoch
                during training. Default is True. It can be set to False to
                reduce the computation cost, if we just care aboud the training
                accuracy or the final accuracy.
        """

    @abc.abstractmethod
    def get_final_accuracy(self) -> tuple[float, float]:
        """Get the final accuracy of the network after training.

        This method may avoid one more accuracy calculation outside.

        Returns:
            tuple[float, float]: Training and test accuracy.
        """

    @abc.abstractmethod
    def get_history_accuracy(self) -> tuple[list[float], list[float]]:
        """Get the history of the accuracy during training.

        Returns:
            tuple[list[float], list[float]]: Training and test accuracy every epoch.
        """
