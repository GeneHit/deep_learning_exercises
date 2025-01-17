import pickle
import time

from ch03_network_forward.c_nn_mnist import Simple3LayerNN
from dataset.mnist import load_mnist


class TestSimple3LayerNN:
    def __init__(self) -> None:
        with open("sample_weight.pkl", "rb") as f:
            self._parameters = pickle.load(f)

        # Load MNIST data, returning a nx784 array for x and a nx10 array for t
        # not necessary to use the traning data.
        _, (self._x_test, self._t_test) = load_mnist(
            normalize=True, flatten=True, one_hot_label=False
        )

    def test_accuracy_with_for_cycle(self) -> None:
        nn = Simple3LayerNN(self._parameters)
        start_time = time.time()
        accuracy = nn.accuracy_with_for_cycle(self._x_test, self._t_test)
        end_time = time.time()
        print(f"Accuracy (for cycle): {accuracy}. ")
        print(f"Elapsed time: {end_time - start_time} seconds.")
        assert 0.9 < accuracy

    def test_accuracy_with_batch(self) -> None:
        nn = Simple3LayerNN(self._parameters)
        start_time = time.time()
        accuracy = nn.accuracy_with_batch(self._x_test, self._t_test)
        end_time = time.time()
        print(f"Accuracy (for cycle): {accuracy}. ")
        print(f"Elapsed time: {end_time - start_time} seconds.")
        assert 0.9 < accuracy
