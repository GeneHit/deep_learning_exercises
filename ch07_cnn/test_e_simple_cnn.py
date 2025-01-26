from ch06_learning_technique.a_optimization import Adam
from ch06_learning_technique.d_trainer import NormalTraier
from ch07_cnn.e_simple_cnn import (
    ConvConfig,
    PoolConfig,
    SimpleCNN,
    SimpleCNNConfig,
)
from dataset.mnist import load_mnist


def test_simple_cnn() -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    # Reduce the dataset size to speed up the test, so that run the test for
    # verifying the correctness of the code.
    # x_train = x_train[:5000]
    # t_train = t_train[:5000]

    epochs = 20

    cnn_params = SimpleCNNConfig(
        input_dim=(1, 28, 28),
        conv_params=ConvConfig(
            filter_num=30, filter_h=5, filter_w=5, stride=1, pad=0
        ),
        pooling_params=PoolConfig(pool_h=2, pool_w=2, stride=2, pad=0),
        hidden_size=100,
        output_size=10,
    )
    nn = SimpleCNN(cnn_params)
    optimizer = Adam(lr=0.01)
    trainer = NormalTraier(
        network=nn,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=epochs,
        mini_batch_size=100,
        evaluated_sample_per_epoch=1000,
    )

    trainer.train()

    train_acc_list, test_acc_list = trainer.get_history_accuracy()
    assert train_acc_list[-1] >= 0.98
    assert test_acc_list[-1] >= 0.95
