"""Weight decay regularization for neural network.
To make we code more modular and fater, we will implement 2 classes
- Sequential2d
- LayerTrainer
which is very helpful and convenient for build a deep neural networks.
The following chapter will use these classes to build a deep neural network.
"""

import pickle
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from common.base import Layer, Optimizer, Trainer
from common.default_type_array import np_float, np_ones

WEIGHT_START_WITH = "W"


class Sequential(Layer):
    """Sequential layers for neural network.

    diagram:
        input - hidden_layers - ouput
    where hidden_layers could be a large number of layers for deep net.
    """

    def __init__(self, layers: tuple[Layer, ...]) -> None:
        self._layers = layers

    def named_params(self) -> dict[str, NDArray[np.floating]]:
        """Return the parameters of the network.

        Note: this return a reference, the dict and the NDArray are mutable.
        It can be used for updating the parameters by outside -=, +=.
        """
        params: dict[str, NDArray[np.floating]] = {}
        for layer in self._layers:
            params.update(layer.named_params())
        return params

    def train(self, flag: bool) -> None:
        """Set the training flag for the network."""
        for layer in self._layers:
            layer.train(flag)

    def forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """See the base class."""
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def backward(self, dout: NDArray[np.floating]) -> NDArray[np.floating]:
        """See the base class."""
        for layer in reversed(self._layers):
            dout = layer.backward(dout)
        return dout

    def param_grads(self) -> dict[str, NDArray[np.floating]]:
        """Return the gradients of the parameters."""
        grads: dict[str, NDArray[np.floating]] = {}
        for layer in self._layers:
            grads.update(layer.param_grads())
        return grads

    def save_params(self, file_name: str) -> None:
        """Save the parameters to a file.

        For the network that is a simple sequential layer.
        """
        params = self.named_params()
        with open(file_name, "wb") as f:
            pickle.dump(params, f)


class LayerTrainer(Trainer):
    """A trainer for training a neural network.

    This is for the network that is based on the Layer class.
    """

    def __init__(
        self,
        network: Layer,
        loss: Layer,
        evaluation_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], float
        ],
        optimizer: Optimizer,
        x_train: NDArray[np.floating],
        t_train: NDArray[np.floating],
        x_test: NDArray[np.floating],
        t_test: NDArray[np.floating],
        epochs: int,
        mini_batch_size: int,
        weight_decay_lambda: float | None = None,
        evaluate_train_data: bool = True,
        evaluate_test_data: bool = True,
        evaluated_sample_per_epoch: int | None = None,
        verbose: bool = False,
        name: str = "",
    ) -> None:
        """Initialize the trainer.

        Parameters:
            network : Layer
                The neural network to be trained.
            loss : Layer
                The loss function to be used for training.
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
            weight_decay_lambda : float | None
                The lambda for the weight decay, using L2 regularization.
            evaluate_train_data : bool
                If True, evaluate the training data.
            evaluate_test_data : bool
                If True, evaluate the test data.
            evaluated_sample_per_epoch : int | None
                Number of samples to evaluate per epoch.
            verbose : bool
                If True, print the training progress.
            name : str
                Name of the trainer, for process bar and logging.
        """
        self._network = network
        self._loss = loss
        self._evaluation_fn = evaluation_fn
        self._optimizer = optimizer
        self._x_train = x_train
        self._t_train = t_train
        self._x_test = x_test
        self._t_test = t_test
        self._epochs = epochs
        self._mini_batch_size = mini_batch_size
        self._weight_decay_lambda: np.floating | None = None
        if weight_decay_lambda is not None:
            self._weight_decay_lambda = np_float(weight_decay_lambda)
        self._evaluate_train_data = evaluate_train_data
        self._evaluate_test_data = evaluate_test_data
        self._evaluated_sample_per_epoch = evaluated_sample_per_epoch
        self._verbose = verbose
        self._name = name

        self._net_params = self._network.named_params()
        self._reset_history()

    def _reset_history(self) -> None:
        self.runnig_losses: list[float] = []
        self.train_acc_history: list[float] = []
        self.test_acc_history: list[float] = []
        self._final_accuracy: tuple[float, float] | None = None

    def train(self) -> None:
        self._reset_history()

        self._network.train(True)
        # tqdm progress bar for epochs
        desc = self._name if self._name else "Training Progress"
        epoch_bar = tqdm(range(self._epochs), desc=desc)
        for epoch in epoch_bar:
            self._train_one_epoch()

            # output the necessary logging if necessary
            self._evaluate_if_necessary(epoch)
        self._network.train(False)

    def get_final_accuracy(self) -> tuple[float, float]:
        """Get the final accuracy of the network after training.

        Have to call the train method before calling this method.

        Returns:
            tuple[float, float]: Train and test accuracy.
        """
        if self._final_accuracy is not None:
            return self._final_accuracy

        self._final_accuracy = (
            self._evaluate(
                x=self._x_train, t=self._t_train, process="Train Acc"
            ),
            self._evaluate(x=self._x_test, t=self._t_test, process="Test Acc"),
        )
        return self._final_accuracy

    def get_history_accuracy(self) -> tuple[list[float], list[float]]:
        """Get the history of the training and test accuracy."""
        return self.train_acc_history, self.test_acc_history

    def _evaluate_if_necessary(self, epoch: int) -> None:
        if not self._evaluate_train_data and not self._evaluate_test_data:
            return

        # set the network to the evaluation mode
        self._network.train(False)
        # output the training accuracy if necessary
        if self._evaluate_train_data:
            x_train_sample, t_train_sample = self._x_train, self._t_train
            if self._evaluated_sample_per_epoch is not None:
                num = self._evaluated_sample_per_epoch
                x_train_sample = self._x_train[:num]
                t_train_sample = self._t_train[:num]

            train_acc = self._evaluate(
                x=x_train_sample,
                t=t_train_sample,
                process=f"Epoch {epoch + 1} Training",
            )
            self.train_acc_history.append(train_acc)

        # output the test accuracy if necessary
        if self._evaluate_test_data:
            x_test_sample, t_test_sample = self._x_test, self._t_test
            if self._evaluated_sample_per_epoch is not None:
                num = self._evaluated_sample_per_epoch
                x_test_sample = self._x_test[:num]
                t_test_sample = self._t_test[:num]
            test_acc = self._evaluate(
                x=x_test_sample,
                t=t_test_sample,
                process=f"Epoch {epoch + 1} Test",
            )
            self.test_acc_history.append(test_acc)

        # set the network to the training mode
        self._network.train(True)

    def _evaluate(
        self, x: NDArray[np.floating], t: NDArray[np.floating], process: str
    ) -> float:
        num_batch = x.shape[0] // self._mini_batch_size
        acc_sum = 0.0
        loss_sum = 0.0
        start_idx = 0
        for _ in range(num_batch):
            end_idx = start_idx + self._mini_batch_size
            idx_range = range(start_idx, end_idx)
            y = self._network.forward(x[idx_range])
            acc_sum += self._evaluation_fn(y, t[idx_range])
            if self._verbose:
                loss_sum += self._loss.forward_to_loss(y, t[idx_range])
            start_idx = end_idx

        acc = acc_sum / num_batch
        if self._verbose:
            print(f"{process}: Acc {acc:.4f}; Loss {loss_sum / num_batch:.4f}")
        return acc

    def _train_one_epoch(self) -> None:
        """Train the network for one epoch.

        Steps every iteration:
            - Get the mini-batch
            - Forward
            - Calculate the loss (with weight decay if necessary)
            - Backward
            - Get the gradient of the parameters (use weight decay if necessary)
            - Update the parameters once
        """
        num_train = self._x_train.shape[0]
        num_batch = num_train // self._mini_batch_size

        loss = 0.0
        for _ in range(num_batch):
            selected_index = np.random.choice(num_train, self._mini_batch_size)
            x_batch = self._x_train[selected_index]
            t_batch = self._t_train[selected_index]

            # forward
            y = self._network.forward(x_batch)
            loss += self._loss.forward_to_loss(y, t_batch)
            if self._weight_decay_lambda:
                weight_decay = _weight_square_sum(self._net_params)
                loss += 0.5 * float(self._weight_decay_lambda * weight_decay)

            # backward
            dout = self._loss.backward(dout=np_ones(shape=(1, 1)))
            self._network.backward(dout)

            # get the gradient of the parameters
            grads = self._network.param_grads()
            _update_weight_decay_if_necessary(
                grads, self._net_params, self._weight_decay_lambda
            )

            # update the parameters
            self._optimizer.one_step(self._net_params, grads)
        self.runnig_losses.append(loss / num_batch)


def _weight_square_sum(params: dict[str, NDArray[np.floating]]) -> np.floating:
    weight_decay = np_float(0)
    for key in params.keys():
        if key.startswith(WEIGHT_START_WITH):
            weight_decay += np.sum(params[key] ** 2)

    return weight_decay


def _update_weight_decay_if_necessary(
    params_grad: dict[str, NDArray[np.floating]],
    params: dict[str, NDArray[np.floating]],
    weight_decay_lambda: np.floating | None,
) -> None:
    if weight_decay_lambda:
        for key in params_grad.keys():
            if key.startswith(WEIGHT_START_WITH):
                params_grad[key] += weight_decay_lambda * params[key]
