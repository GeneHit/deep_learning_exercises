import abc

import numpy as np
from numpy.typing import NDArray


class Layer(abc.ABC):
    """Base class for neural network layers."""

    @abc.abstractmethod
    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """

    def train_flag(self, flag: bool) -> None:
        """Set the training flag of the layer.

        During training, some layer may need to change their behavior, for
        example, dropout layer.

        Parameters:
            flag : bool
                Training flag. During training, the flag cann be set to True
                if neccesary.
        """
        pass

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


class NueralNet(abc.ABC):
    @abc.abstractmethod
    def named_parameters(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters outside.
        """

    @abc.abstractmethod
    def predict(
        self, x: NDArray[np.floating], train_flag: bool = False
    ) -> NDArray[np.floating]:
        """Predict the output for the given input.

        Parameters:
            x (NDArray[np.floating]): Input data.
            train_flag (bool): Training flag.

        Returns:
            NDArray[np.floating]: Predicted output.
        """

    @abc.abstractmethod
    def loss(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
        train_flag: bool = False,
    ) -> float:
        """Calculate the loss for the given input and target output.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.
            train_flag (bool): Training flag.

        Returns:
            float: Loss value.
        """

    @abc.abstractmethod
    def accuracy(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
    ) -> float:
        """Calculate the accuracy for the given input and target output.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            float: Accuracy value.
        """

    @abc.abstractmethod
    def gradient(
        self,
        x: NDArray[np.floating],
        t: NDArray[np.floating],
    ) -> dict[str, NDArray[np.floating]]:
        """Calculate the gradient of the parameters.

        Parameters:
            x (NDArray[np.floating]): Input data.
            t (NDArray[np.floating]): Target output.

        Returns:
            dict[str, NDArray[np.floating]]: Gradients of the weights and biases.
        """


class Optimizer(abc.ABC):
    """Base class for all optimizers."""

    @abc.abstractmethod
    def update(
        self,
        params: dict[str, NDArray[np.floating]],
        grads: dict[str, NDArray[np.floating]],
    ) -> None:
        """Update the parameters using the gradients.

        Parameters:
            params : dict[str, NDArray[np.floating]]
                Dictionary of parameters to be updated.
            grads : dict[str, NDArray[np.floating]]
                Dictionary of gradients for the parameters.
        """


class Trainer(abc.ABC):
    """A class for training a neural network."""

    def __init__(
        self,
        network: NueralNet,
        optimizer: Optimizer,
        x_train: NDArray[np.floating],
        t_train: NDArray[np.floating],
        x_test: NDArray[np.floating],
        t_test: NDArray[np.floating],
        epochs: int,
        mini_batch_size: int,
        evaluate_train_data: bool = True,
        evaluate_test_data: bool = True,
        evaluated_sample_per_epoch: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the trainer.

        Parameters:
            network : NueralNet
                The neural network to be trained.
            optimizer : Optimizer
                The optimizer to be used for training.
            x_train : NDArray[np.floating]
                Training data.
            t_train : NDArray[np.floating]
                Training labels.
            x_test : NDArray[np.floating]
                Test data.
            t_test : NDArray[np.floating]
                Test labels.
            epochs : int
                Number of epochs.
            mini_batch_size : int
                Mini-batch size.
            evaluate_train_data : bool
                Evaluate the training data during training, by default True.
            evaluate_test_data : bool
                Evaluate the test data during training, by default True.
            evaluated_sample_per_epoch : int, optional
                Number of test samples for evaluation per epoch, by default None.
            verbose : bool
                Print the training logging, by default False.
        """
        raise NotImplementedError("have to implement the __init__ method.")

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
        raise NotImplementedError("The train method is not implemented yet.")

    @abc.abstractmethod
    def get_final_accuracy(self) -> float:
        """Get the final accuracy of the network after training.

        This method may avoid one more accuracy calculation outside.
        """
        raise NotImplementedError

    def get_history_accuracy(self) -> tuple[list[float], list[float]]:
        """Get the history of the accuracy during training.

        Returns:
            tuple[list[float], list[float]]: Training and test accuracy every epoch.
        """
        raise NotImplementedError
