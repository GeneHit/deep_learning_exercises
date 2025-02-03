import numpy as np
import pytest
from numpy.typing import NDArray

from ch06_learning_technique.a_optimization import SGD
from ch06_learning_technique.d_reg_weight_decay import LayerTraier
from ch06_learning_technique.test_d_overfit_exp import _plot_accuracy
from common.evaluation import single_label_accuracy
from common.layer_config import (
    AffineConfig,
    BatchNorm1dConfig,
    ReLUConfig,
    SequentialConfig,
    SoftmaxWithLossConfig,
)
from common.utils import assert_layer_parameter_type
from dataset.mnist import load_mnist

EPOCHS = 20
HIDDEN_SIZES = (100, 100, 100, 100, 100)


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
) -> LayerTraier:
    """Train a overfitting NN without batch normalization and dropout.

    This is used for comparing the overfitting with weight decay and dropout.
    """
    # returning a 60000x784 array for x and a nx10 array for t
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
            AffineConfig(in_size=100, out_size=10, param_suffix="5"),
        ),
    )
    network = config.create()
    optimizer = SGD(lr=0.01)
    trainer = LayerTraier(
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
        evaluate_test_data=False,
        name="OverfitNN",
    )

    # Train the network
    trainer.train()
    assert_layer_parameter_type(network)

    return trainer


def test_batch_normalization(
    overfit_nn: LayerTraier,
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> None:
    """Evaluate the batch normalization."""
    # returning a 1000x784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, param_suffix="1"),
            BatchNorm1dConfig(num_feature=100, param_suffix="1"),
            # use inplace to save memory avoiding to create a new array
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=100, param_suffix="2"),
            BatchNorm1dConfig(num_feature=100, param_suffix="2"),
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=100, param_suffix="3"),
            BatchNorm1dConfig(num_feature=100, param_suffix="3"),
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=100, param_suffix="4"),
            BatchNorm1dConfig(num_feature=100, param_suffix="4"),
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=10, param_suffix="5"),
        ),
    )
    network = config.create()
    optimizer = SGD(lr=0.01)
    trainer = LayerTraier(
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
        evaluate_test_data=False,
        name="BatchNormExp",
    )

    # Train the network
    trainer.train()
    assert_layer_parameter_type(network)

    train_acc_list, _ = trainer.get_history_accuracy()
    (overfit_train_acc_list, _) = overfit_nn.get_history_accuracy()
    # Set to True to plot the accuracy history for comparison
    plot_data = False
    if plot_data:
        _plot_accuracy(
            train_acc_list,
            [],
            overfit_train_acc_list,
            [],
            showing_time=None,
        )
    assert train_acc_list[-1] > overfit_train_acc_list[-1]
