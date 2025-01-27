import numpy as np
from numpy.typing import NDArray

from common.base import NueralNet, Optimizer, TrainerBase


class NormalTraier(TrainerBase):
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
        evaluate_test_data: bool = True,
        evaluated_sample_per_epoch: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._network = network
        self._optimizer = optimizer
        self._x_train = x_train
        self._t_train = t_train
        self._x_test = x_test
        self._t_test = t_test
        self._epochs = epochs
        self._mini_batch_size = mini_batch_size
        self._evaluated_sample_per_epoch = evaluated_sample_per_epoch
        self._verbose = verbose

    def train(self) -> None:
        raise NotImplementedError

    def get_final_accuracy(self) -> tuple[float, float]:
        raise NotImplementedError

    def get_history_accuracy(self) -> tuple[list[float], list[float]]:
        raise NotImplementedError
