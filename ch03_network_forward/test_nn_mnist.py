import pickle
import time

import numpy as np
import pytest

from ch03_network_forward.c_nn_mnist import Simple3LayerNN
from dataset.mnist import load_mnist


@pytest.fixture(scope="module")
def parameters() -> dict[str, np.typing.NDArray[np.floating]]:
    with open("ch03_network_forward/sample_weight.pkl", "rb") as f:
        # for mypy
        params: dict[str, np.typing.NDArray[np.floating]] = pickle.load(f)
        return params


@pytest.fixture(scope="module")
def mnist_test_data() -> tuple[
    np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]
]:
    _, (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


class TestSimple3LayerNN:
    def test_accuracy_with_for_cycle(
        self,
        parameters: dict[str, np.typing.NDArray[np.floating]],
        mnist_test_data: tuple[
            np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]
        ],
    ) -> None:
        nn = Simple3LayerNN(parameters)
        start_time = time.time()
        x_test, t_test = mnist_test_data
        accuracy = nn.accuracy_with_for_cycle(x_test, t_test)
        end_time = time.time()
        print(f"Accuracy (for cycle): {accuracy}. ")
        print(f"Elapsed time: {end_time - start_time} seconds.")
        assert 0.9 < accuracy

    def test_accuracy_with_batch(
        self,
        parameters: dict[str, np.typing.NDArray[np.floating]],
        mnist_test_data: tuple[
            np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]
        ],
    ) -> None:
        nn = Simple3LayerNN(parameters)
        start_time = time.time()
        x_test, t_test = mnist_test_data
        accuracy = nn.accuracy_with_batch(x_test, t_test)
        end_time = time.time()
        print(f"Accuracy (for cycle): {accuracy}. ")
        print(f"Elapsed time: {end_time - start_time} seconds.")
        assert 0.9 < accuracy
