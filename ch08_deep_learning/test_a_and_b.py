from ch06_learning_technique.a_optimization import Adam
from ch06_learning_technique.d_trainer import NormalTraier
from ch07_cnn.e_simple_cnn import ConvConfig, PoolConfig
from ch08_deep_learning.a_deep_network import DeepConvNet, DeepConvNetConfig
from ch08_deep_learning.b_data_augmentation import augment_mnist_data
from dataset.mnist import load_mnist


def test_deep_conv_net() -> None:
    # set save_params to True to save the parameters
    _test_deep_cnn(save_params=False)


def test_deep_conv_net_with_data_augmentation() -> None:
    # set save_params to True to save the parameters
    _test_deep_cnn(augmente_data=True, save_params=False)


def _test_deep_cnn(
    augmente_data: bool = False, save_params: bool = False
) -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    if augmente_data:
        x_train, t_train = augment_mnist_data(
            x_train, t_train, augmentation_factor=1.5
        )

    net_config = DeepConvNetConfig(
        input_dim=(1, 28, 28),
        hidden_layer_configs=(
            ConvConfig(filter_num=16, filter_h=3, filter_w=3, stride=1, pad=1),
            ConvConfig(filter_num=16, filter_h=3, filter_w=3, stride=1, pad=1),
            PoolConfig(pool_h=2, pool_w=2, stride=2, pad=0),
            ConvConfig(filter_num=32, filter_h=3, filter_w=3, stride=1, pad=1),
            ConvConfig(filter_num=32, filter_h=3, filter_w=3, stride=1, pad=2),
            PoolConfig(pool_h=2, pool_w=2, stride=2, pad=0),
            ConvConfig(filter_num=64, filter_h=3, filter_w=3, stride=1, pad=1),
            ConvConfig(filter_num=64, filter_h=3, filter_w=3, stride=1, pad=1),
            PoolConfig(pool_h=2, pool_w=2, stride=2, pad=0),
        ),
        hidden_size=50,
        output_size=10,
    )
    network = DeepConvNet(config=net_config)
    optimizer = Adam(lr=0.001)
    trainer = NormalTraier(
        network=network,
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
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    print(f"train acc, test acc | {train_acc:.4f}, {test_acc:.4f}")
    assert train_acc >= 0.99
    assert test_acc >= 0.99

    if save_params:
        network.save_params("deep_convnet_params.pkl")
        print("Saved Network Parameters!")
