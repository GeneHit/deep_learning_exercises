from ch06_learning_technique.a_optimization import Adam
from ch06_learning_technique.d_reg_weight_decay import LayerTrainer
from common.evaluation import single_label_accuracy
from common.layer_config import (
    AffineConfig,
    Conv2dConfig,
    FlattenConfig,
    MaxPool2dConfig,
    ReLUConfig,
    SequentialConfig,
    SoftmaxWithLossConfig,
)
from common.utils import assert_layer_parameter_type
from dataset.mnist import load_mnist


def test_simple_cnn() -> None:
    # returning a nx1x28x28 array for x and a nx10 array for t
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    # Reduce the dataset size to speed up the test, so that run the test for
    # verifying the correctness of the code.
    x_train = x_train[:5000]
    t_train = t_train[:5000]
    epochs = 20

    # Initializing the network:
    #   (conv - relu - max_pool) - (flatten - affine - relu) - affine - softmax
    config = SequentialConfig(
        # input_dim=(1, 28, 28),
        hidden_layer_configs=(
            Conv2dConfig(
                in_channels=1,
                out_channels=30,
                kernel_size=(5, 5),
                stride=1,
                pad=0,
                param_suffix="1",
            ),
            ReLUConfig(),
            MaxPool2dConfig(kernel_size=(2, 2), stride=2),
            FlattenConfig(),
            AffineConfig(in_size=30 * 12 * 12, out_size=100, param_suffix="2"),
            ReLUConfig(),
            AffineConfig(in_size=100, out_size=10, param_suffix="3"),
        ),
    )
    network = config.create()
    optimizer = Adam(lr=0.01)
    trainer = LayerTrainer(
        network=network,
        loss=SoftmaxWithLossConfig().create(),
        evaluation_fn=single_label_accuracy,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=epochs,
        mini_batch_size=99,
        evaluate_train_data=False,
        evaluate_test_data=False,
    )

    trainer.train()

    assert_layer_parameter_type(network)
    train_acc_list, test_acc_list = trainer.get_final_accuracy()
    assert train_acc_list >= 0.98
    assert test_acc_list >= 0.95
