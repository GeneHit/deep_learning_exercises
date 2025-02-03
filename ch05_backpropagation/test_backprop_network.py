from ch04_network_learning.d_learning_implementation import training
from ch05_backpropagation.c_backprop_network import BackPropTwoLayerNN
from common.default_type_array import get_default_type, np_float
from common.utils import log_duration
from dataset.mnist import load_mnist


def test_two_layer_nn() -> None:
    # Load MNIST data, returning a nx784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = load_mnist(
        normalize=True, flatten=True, one_hot_label=True
    )

    # Initialize the neural network
    network = BackPropTwoLayerNN(input_size=784, hidden_size=50, output_size=10)

    # Train the network (assuming a train method has existed)
    # The train method should return the latest training history losses
    with log_duration("training"):
        training(
            network,
            x_train,
            t_train,
            batch_size=99,
            learning_rate=np_float(0.1),
            epochs=20,
            verbose=True,
        )

    for key, value in network.named_parameters().items():
        assert value.dtype == get_default_type(), f"key: {key}"
    # Get the accuracy using the network's accuracy method
    accuracy = network.accuracy(x_test, t_test)

    # Check that the latest training loss is less than 2.5
    # and accuracy is greater than 0.9
    assert accuracy > 0.9
