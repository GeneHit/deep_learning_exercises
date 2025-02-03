import numpy as np

from ch04_network_learning.d_learning_implementation import TwoLayerNN, training
from common.default_type_array import np_float
from common.utils import log_duration
from dataset.mnist import load_mnist


def test_training() -> None:
    # Load MNIST data, returning a nx784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = load_mnist(
        normalize=True, flatten=True, one_hot_label=True
    )
    # about 26 seconds for 1 batch
    batch_size = 100
    # do not verify the accuracy, just verify the code runs
    # because the numerical gradient is svery low, we use a small dataset
    train_num = 100
    epoch = 1
    target_accuracy: None | float = None

    # a test parameters for higher accuracy, it takes a long time (> 1h).
    # train_num = x_train.shape[0]  # 60000
    # epoch = 20
    # target_accuracy = 0.9

    # shuffle the data
    idx = np.random.choice(x_train.shape[0], train_num)
    x_train = x_train[idx]
    t_train = t_train[idx]

    # Initialize the neural network
    network = TwoLayerNN(input_size=784, hidden_size=50, output_size=10)

    # Train the network using the training function
    # The training function should return the latest training loss
    with log_duration("training"):
        training(
            network,
            x_train,
            t_train,
            batch_size=batch_size,
            learning_rate=np_float(0.1),
            epochs=epoch,
            verbose=False,
        )

    # Get the accuracy using the network's accuracy method
    accuracy = network.accuracy(x_test, t_test)

    if target_accuracy is not None:
        assert accuracy > target_accuracy
