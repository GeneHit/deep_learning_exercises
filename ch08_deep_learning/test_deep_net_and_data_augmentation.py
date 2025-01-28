from ch06_learning_technique.a_optimization import Adam
from ch06_learning_technique.d_reg_weight_decay import LayerTraier, Sequential
from ch08_deep_learning.a_data_augmentation import augment_mnist_data
from common.evaluation import single_label_accuracy
from common.layer_config import (
    AffineConfig,
    Conv2dConfig,
    Dropout2dConfig,
    MaxPool2dConfig,
    ReLUConfig,
    SequentialConfig,
    SoftmaxWithLossConfig,
)
from dataset.mnist import load_mnist


def test_deep_conv_net() -> None:
    # set save_params to True to save the parameters
    _test_deep_conv_net(save_params=False)


def test_deep_conv_net_with_data_augmentation() -> None:
    # set save_params to True to save the parameters
    _test_deep_conv_net(augmente_data=True, save_params=False)


def _test_deep_conv_net(
    augmente_data: bool = False, save_params: bool = False
) -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    if augmente_data:
        x_train, t_train = augment_mnist_data(
            x_train, t_train, augmentation_factor=1.5
        )

    # diagram:
    #     conv - relu - conv- relu - max_pool -
    #     conv - relu - conv- relu - max_pool -
    #     conv - relu - conv- relu - max_pool -
    #     affine - relu - dropout - affine - dropout - softmax
    net_config = SequentialConfig(
        # input_dim=(1, 28, 28),
        hidden_layer_configs=(
            Conv2dConfig(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                pad=1,
                param_suffix="1",
            ),
            ReLUConfig(),
            Conv2dConfig(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                pad=1,
                param_suffix="2",
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2, pad=0),
            Conv2dConfig(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                pad=1,
                param_suffix="3",
            ),
            ReLUConfig(),
            Conv2dConfig(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                pad=2,
                param_suffix="4",
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2, pad=0),
            Conv2dConfig(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                pad=1,
                param_suffix="5",
            ),
            ReLUConfig(),
            Conv2dConfig(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                pad=1,
                param_suffix="6",
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2, pad=0),
            AffineConfig(in_size=64 * 3 * 3, out_size=50, param_suffix="7"),
            ReLUConfig(),
            Dropout2dConfig(dropout_ratio=0.5),
            AffineConfig(in_size=50, out_size=10, param_suffix="8"),
            Dropout2dConfig(dropout_ratio=0.5),
        ),
        # load_params="deep_2d_net_params.pkl",
    )
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
    trainer.train()

    # Check the final accuracy
    train_acc, test_acc = trainer.get_final_accuracy()
    print(f"train acc, test acc | {train_acc:.4f}, {test_acc:.4f}")
    assert train_acc >= 0.99
    assert test_acc >= 0.99

    if save_params:
        assert isinstance(network, Sequential)  # for mypy
        network.save_params("deep_2d_net_params.pkl")
        print("Saved Network Parameters!")
