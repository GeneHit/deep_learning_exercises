from typing import Callable

import numpy as np
from numpy.typing import NDArray

from common.base import Layer, Optimizer, TrainerBase
from common.utils import update_weight_decay_if_necessary, weight_square_sum


class LayerTraier(TrainerBase):
    """A trainer for training a neural network.

    This is for the network that is based on the Layer class.
    """

    def __init__(
        self,
        network: Layer,
        loss: Layer,
        evaluation_fn: Callable[
            [Layer, NDArray[np.floating], NDArray[np.floating]], float
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
            evaluate_train_data : bool
                If True, evaluate the training data.
            evaluate_test_data : bool
                If True, evaluate the test data.
            evaluated_sample_per_epoch : int | None
                Number of samples to evaluate per epoch.
            verbose : bool
                If True, print the training progress.
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
        self._weight_decay_lambda = weight_decay_lambda
        self._evaluate_train_data = evaluate_train_data
        self._evaluate_test_data = evaluate_test_data
        self._evaluated_sample_per_epoch = evaluated_sample_per_epoch
        self._verbose = verbose

        self._net_params = self._network.named_params()
        self._train_acc_history: list[float] = []
        self._test_acc_history: list[float] = []
        self._final_accuracy: tuple[float, float] | None = None

    def train(self) -> None:
        self._train_acc_history = []
        self._test_acc_history = []

        self._network.train(True)
        # TODO: use tqdm to show the progress
        for epoch in range(self._epochs):
            print(f"Epoch {epoch + 1}/{self._epochs}")
            self._train_one_epoch()

            # output the necessary logging if necessary
            self._evaluate_if_necessary(epoch)

    def get_final_accuracy(self) -> tuple[float, float]:
        """Get the final accuracy of the network after training.

        Have to call the train method before calling this method.

        Returns:
            tuple[float, float]: Train and test accuracy.
        """
        if self._final_accuracy is not None:
            return self._final_accuracy

        def get_final_test_accuracy() -> float:
            if (
                self._evaluate_test_data
                and self._evaluated_sample_per_epoch is None
            ):
                return self._test_acc_history[-1]
            return self._evaluation_fn(
                self._network, self._x_test, self._t_test
            )

        def get_final_train_accuracy() -> float:
            if (
                self._evaluate_train_data
                and self._evaluated_sample_per_epoch is None
            ):
                return self._train_acc_history[-1]
            return self._evaluation_fn(
                self._network, self._x_train, self._t_train
            )

        self._final_accuracy = (
            get_final_train_accuracy(),
            get_final_test_accuracy(),
        )
        return self._final_accuracy

    def get_history_accuracy(self) -> tuple[list[float], list[float]]:
        """Get the history of the training and test accuracy."""
        return self._train_acc_history, self._test_acc_history

    def _train_one_epoch(self) -> None:
        iterations_pre_epoch = len(self._x_train) // self._mini_batch_size
        for _ in range(iterations_pre_epoch):
            # get mini-batch
            selected_index = np.random.choice(
                self._x_train.shape[0], self._mini_batch_size
            )
            x_batch = self._x_train[selected_index]
            t_batch = self._t_train[selected_index]

            # forward
            y = self._network.forward(x_batch)
            loss = self._loss.forward_to_loss(y, t_batch)
            if self._weight_decay_lambda:
                loss += (
                    0.5
                    * self._weight_decay_lambda
                    * (weight_square_sum(self._net_params))
                )

            # backward
            d_out = self._loss.backward(dout=np.ones_like(loss))
            self._network.backward(d_out)

            # get the gradient of the parameters
            params_grad = self._network.param_grads()
            update_weight_decay_if_necessary(
                params_grad, self._net_params, self._weight_decay_lambda
            )

            # update the parameters
            self._optimizer.update(self._net_params, params_grad)

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

            train_accuracy = self._evaluation_fn(
                self._network, x_train_sample, t_train_sample
            )
            self._train_acc_history.append(train_accuracy)
            if self._verbose:
                print(
                    f"Train Accuracy at epoch {epoch + 1}: {train_accuracy:.4f}"
                )

        # output the test accuracy if necessary
        if self._evaluate_test_data:
            x_test_sample, t_test_sample = self._x_test, self._t_test
            if self._evaluated_sample_per_epoch is not None:
                num = self._evaluated_sample_per_epoch
                x_test_sample = self._x_test[:num]
                t_test_sample = self._t_test[:num]
            test_accuracy = self._evaluation_fn(
                self._network, x_test_sample, t_test_sample
            )
            self._test_acc_history.append(test_accuracy)
            if self._verbose:
                print(
                    f"Test Accuracy at epoch {epoch + 1}: {test_accuracy:.4f}"
                )

        # set the network to the training mode
        self._network.train(True)
