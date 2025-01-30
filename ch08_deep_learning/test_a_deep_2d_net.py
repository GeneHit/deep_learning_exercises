import time
from typing import TypeAlias

from ch06_learning_technique.a_optimization import Adam
from ch06_learning_technique.d_reg_weight_decay import LayerTraier, Sequential
from ch08_deep_learning.a_data_augmentation import augment_mnist_data
from ch08_deep_learning.a_deep_2d_net import deep_2d_net_config
from common.default_type_array import get_default_type, set_default_type
from common.evaluation import single_label_accuracy
from common.layer_config import (
    SequentialConfig,
    SoftmaxWithLossConfig,
)
from common.utils import assert_layer_parameter_type
from dataset.mnist import load_mnist


def test_deep_conv_net() -> None:
    # set save_params to True to save the parameters
    train_and_test_deep_conv_net(
        net_config=deep_2d_net_config(), save_params=False
    )


def train_and_test_deep_conv_net(
    net_config: SequentialConfig,
    augmente_data: bool = False,
    save_params: bool = False,
    target_float_dtype: TypeAlias = get_default_type(),
) -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    if augmente_data:
        x_train, t_train = augment_mnist_data(
            x_train, t_train, augmentation_factor=1.5
        )

    set_default_type(target_float_dtype)
    network = net_config.create()
    optimizer = Adam(lr=0.001)
    trainer = LayerTraier(
        network=network,
        loss=SoftmaxWithLossConfig().create(),
        evaluation_fn=single_label_accuracy,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=20,
        mini_batch_size=100,
        evaluated_sample_per_epoch=1000,
    )

    start_time = time.time()

    trainer.train()

    duration = time.time() - start_time
    print(f"Training duration: {duration:.2f} seconds")

    assert_layer_parameter_type(network)

    # Check the final accuracy
    train_acc, test_acc = trainer.get_final_accuracy()
    print(f"train acc, test acc | {train_acc:.4f}, {test_acc:.4f}")
    assert train_acc >= 0.99
    assert test_acc >= 0.99

    if save_params:
        assert isinstance(network, Sequential)  # for mypy
        network.save_params("deep_2d_net_params.pkl")
        print("Saved Network Parameters!")
