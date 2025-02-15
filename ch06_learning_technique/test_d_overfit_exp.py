import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from ch06_learning_technique.a_optimization import SGD
from ch06_learning_technique.d_reg_weight_decay import LayerTrainer
from common.evaluation import single_label_accuracy
from common.layer_config import (
    AffineConfig,
    DropoutConfig,
    ReLUConfig,
    SequentialConfig,
    SoftmaxWithLossConfig,
)
from common.utils import assert_layer_parameter_type
from dataset.mnist import load_mnist

EPOCHS = 200


# Use the module scope to load the MNIST data only once, then share it across
@pytest.fixture(scope="module")
def mnist_data() -> tuple[
    tuple[NDArray[np.floating], NDArray[np.floating]],
    tuple[NDArray[np.floating], NDArray[np.floating]],
]:
    """Return the MNIST data.

    Loads the MNIST dataset, normalizes the pixel values, and converts the
    labels to one-hot encoding.
    Returns:
        tuple: Tuple containing training and test data.
    """
    # Load MNIST data, returning a 60000x784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = load_mnist(
        normalize=True, flatten=True
    )
    # Use only 300 samples for testing the overfitting
    return (x_train[:300], t_train[:300]), (x_test, t_test)


@pytest.fixture(scope="module")
def overfit_nn(
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> LayerTrainer:
    """Train a overfitting NN without batch normalization and dropout.

    This is used for comparing the overfitting with weight decay and dropout.
    """
    # returning a 300X784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, param_suffix="1"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="2"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="3"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="4"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="5"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="6"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=10, param_suffix="7"),
        ),
    )
    network = config.create()
    optimizer = SGD(lr=0.01)
    trainer = LayerTrainer(
        network=network,
        loss=SoftmaxWithLossConfig().create(),
        evaluation_fn=single_label_accuracy,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=EPOCHS,
        mini_batch_size=99,
        name="OverfitNN",
        verbose=False,
    )
    # Train the network
    trainer.train()

    assert_layer_parameter_type(network)

    return trainer


def test_overfit_with_weight_decay(
    overfit_nn: LayerTrainer,
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> None:
    """Verify that the NN with weight decay can reduce the overfitting."""
    # returning a 300X784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, param_suffix="1"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="2"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="3"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="4"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="5"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, param_suffix="6"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=10, param_suffix="7"),
        ),
    )
    network = config.create()
    optimizer = SGD(lr=0.01)
    trainer = LayerTrainer(
        network=network,
        loss=SoftmaxWithLossConfig().create(),
        evaluation_fn=single_label_accuracy,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=EPOCHS,
        mini_batch_size=99,
        weight_decay_lambda=0.1,
        name="WeightDecayExp",
        verbose=False,
    )

    # Train the network
    trainer.train()

    assert_layer_parameter_type(network)
    train_acc_list, test_acc_list = trainer.get_history_accuracy()
    (overfit_train_acc_list, overfit_test_acc_list) = (
        overfit_nn.get_history_accuracy()
    )
    # Set to True to plot the accuracy history for comparison
    plot_data = False
    if plot_data:
        _plot_accuracy(
            train_acc_list,
            test_acc_list,
            overfit_train_acc_list,
            overfit_test_acc_list,
            showing_time=None,
        )
    # the weight decay should reduce the overfitting
    assert (train_acc_list[-1] - test_acc_list[-1]) < (
        overfit_train_acc_list[-1] - overfit_test_acc_list[-1]
    )


def test_overfit_with_dropout(
    overfit_nn: LayerTrainer,
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> None:
    """Verify the implementation of multi layer NN with weight decay."""
    # returning a 300X784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    dropout_ratio = 0.2
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, param_suffix="1"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=dropout_ratio),
            AffineConfig(in_size=100, out_size=100, param_suffix="2"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=dropout_ratio),
            AffineConfig(in_size=100, out_size=100, param_suffix="3"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=dropout_ratio),
            AffineConfig(in_size=100, out_size=100, param_suffix="4"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=dropout_ratio),
            AffineConfig(in_size=100, out_size=100, param_suffix="5"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=dropout_ratio),
            AffineConfig(in_size=100, out_size=100, param_suffix="6"),
            ReLUConfig(),
            DropoutConfig(dropout_ratio=dropout_ratio),
            AffineConfig(in_size=100, out_size=10, param_suffix="7"),
        ),
    )
    network = config.create()
    optimizer = SGD(lr=0.01)
    trainer = LayerTrainer(
        network=network,
        loss=SoftmaxWithLossConfig().create(),
        evaluation_fn=single_label_accuracy,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=EPOCHS,
        mini_batch_size=100,
        name="DropoutExp",
    )

    # Train the network
    trainer.train()
    assert_layer_parameter_type(network)

    train_acc_list, test_acc_list = trainer.get_history_accuracy()
    (overfit_train_acc_list, overfit_test_acc_list) = (
        overfit_nn.get_history_accuracy()
    )
    # Set to True to plot the accuracy history for comparison
    plot_data = False
    if plot_data:
        _plot_accuracy(
            train_acc_list,
            test_acc_list,
            overfit_train_acc_list,
            overfit_test_acc_list,
            showing_time=None,
        )
    # the dropout should reduce the overfitting
    assert (train_acc_list[-1] - test_acc_list[-1]) < (
        overfit_train_acc_list[-1] - overfit_test_acc_list[-1]
    )


def _plot_accuracy(
    train_acc_list: list[float],
    test_acc_list: list[float],
    overfit_train_acc_list: list[float],
    overfit_test_acc_list: list[float],
    showing_time: float | None = 2.0,
) -> None:
    """Plot the accuracy history."""
    x = np.arange(len(train_acc_list))
    if train_acc_list:
        plt.plot(x, train_acc_list, marker="o", label="train")
    if test_acc_list:
        plt.plot(x, test_acc_list, marker="s", label="test")
    if overfit_train_acc_list:
        plt.plot(
            x,
            overfit_train_acc_list,
            marker="o",
            linestyle="--",
            label="overfit train",
        )
    if overfit_test_acc_list:
        plt.plot(
            x,
            overfit_test_acc_list,
            marker="s",
            linestyle="--",
            label="overfit test",
        )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    if showing_time is not None:
        plt.pause(showing_time)  # Show the figure for 2 seconds
        plt.close()  # Close the figure properly
    else:
        plt.show()
