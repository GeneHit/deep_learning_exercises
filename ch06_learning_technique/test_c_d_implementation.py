import numpy as np
import pytest
from numpy.typing import NDArray

from ch06_learning_technique.a_optimization import SGD
from ch06_learning_technique.d_reg_weight_decay import LayerTraier
from common.evaluation import single_label_accuracy
from common.layer_config import (
    AffineConfig,
    BatchNorm1dConfig,
    DropoutConfig,
    ReLUConfig,
    SequentialConfig,
    SoftmaxWithLossConfig,
)
from dataset.mnist import load_mnist

# since the accuracy is not always the same and just used for verifying the
# implementation, we can use a lower threshold for less testing time.
ACCURACY_THRESHOLD = 0.5
# there are 60000 training samples, and the batch size is 100. Referencing
# the accuracy threshold, we can set a epochs for less computation.
EPOCHS = 20


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
    return load_mnist(normalize=True, flatten=True)


def test_multi_layer_nn(
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> None:
    """Verify the implementation of multi layer NN.

    It will not use the batch normalization and dropout.
    """
    # returning a 60000x784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=10, initializer="he_normal"),
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
        weight_decay_lambda=0.1,
    )

    # Train the network
    trainer.train()

    # accuracy should be greater than a verifying threshold
    assert any(np.array(trainer.get_final_accuracy()) > ACCURACY_THRESHOLD)


def test_batch_normalization_by_multi_layer_nn(
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> None:
    """Verify the batch normalization implementation."""
    # returning a 60000x784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, initializer="he_normal"),
            BatchNorm1dConfig(num_feature=100),
            # use inplace to save memory avoiding to create a new array
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            BatchNorm1dConfig(num_feature=100),
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            BatchNorm1dConfig(num_feature=100),
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            BatchNorm1dConfig(num_feature=100),
            ReLUConfig(inplace=True),
            AffineConfig(in_size=100, out_size=10, initializer="he_normal"),
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
    )

    # Train the network
    trainer.train()

    # accuracy should be greater than a verifying threshold
    assert any(np.array(trainer.get_final_accuracy()) > ACCURACY_THRESHOLD)


def test_dropout_by_multi_layer_nn(
    mnist_data: tuple[
        tuple[NDArray[np.floating], NDArray[np.floating]],
        tuple[NDArray[np.floating], NDArray[np.floating]],
    ],
) -> None:
    """Verify the dropout implementation."""
    # returning a 60000x784 array for x and a nx10 array for t
    ((x_train, t_train), (x_test, t_test)) = mnist_data

    # Initialization
    config = SequentialConfig(
        # input_dim=(784,),
        hidden_layer_configs=(
            AffineConfig(in_size=784, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            DropoutConfig(),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            DropoutConfig(),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            DropoutConfig(),
            AffineConfig(in_size=100, out_size=100, initializer="he_normal"),
            ReLUConfig(),
            DropoutConfig(),
            AffineConfig(in_size=100, out_size=10, initializer="he_normal"),
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
    )

    # Train the network
    trainer.train()

    # accuracy should be greater than a verifying threshold
    assert any(np.array(trainer.get_final_accuracy()) > ACCURACY_THRESHOLD)
