import numpy as np
from numpy.typing import NDArray

from ch06_learning_technique.a_optimization import SGD
from ch06_learning_technique.d_reg_weight_decay import LayerTraier
from common.evaluation import single_label_accuracy
from common.layer_config import (
    AffineConfig,
    ReLUConfig,
    SequentialConfig,
    SoftmaxWithLossConfig,
)


def hyper_parameter_optimization(
    optimization_trial: int,
    x_train: NDArray[np.floating],
    t_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    t_test: NDArray[np.floating],
    weight_decay_log_bounds: tuple[int, int],
    learning_rate_log_bounds: tuple[int, int],
    epochs: int,
    mini_batch_size: int,
    verbose: bool = False,
) -> dict[str, tuple[list[float], list[float]]]:
    """Optimize hyperparameters using random search.

    Write a hyper-parameter optimization for the MultiLinearNN in the
    ch06_learning_technique/d_reg_weight_decay.py, to find the best weight decay
    and learning rate for the model.

    Parameters:
        optimization_trial (int): Number of optimization trials
        x_train (NDArray[np.floating]): Training data.
        t_train (NDArray[np.floating]): Training target.
        x_test (NDArray[np.floating]): Test data.
        t_test (NDArray[np.floating]): Test target.
        weight_decay_log_bounds : tuple[int, int])
            Bounds for weight decay, like (-8, -4), which means the weight decay
            will be sampled from 1e-8 to 1e-4.
        learning_rate_log_bounds : tuple[int, int])
            Bounds for learning rate, like (-6, -2), which means the learning
            rate will be sampled from 1e-6 to 1e-2.
        epochs (int): Number of epochs.
        mini_batch_size (int): Batch size.
        verbose (bool): Verbosity.

    Returns:
        dict[str, tuple[list[float], list[float]]]:
            Dictionary of hyperparameters and train/val accuracy history.
            str: hyperparameters like "lr:0.001, weight decay:0.001"
            tuple[list[float], list[float]]: Train/validation accuracy history.
    """
    result: dict[str, tuple[list[float], list[float]]] = {}
    for i in range(optimization_trial):
        # Randomly sample the hyperparameters
        weight_decay = 10 ** np.random.uniform(*weight_decay_log_bounds)
        lr = 10 ** np.random.uniform(*learning_rate_log_bounds)

        config = _get_nn_config()
        network = config.create()

        # Train the network
        trainer = LayerTraier(
            network=network,
            loss=SoftmaxWithLossConfig().create(),
            evaluation_fn=single_label_accuracy,
            optimizer=SGD(lr=lr),
            x_train=x_train,
            t_train=t_train,
            x_test=x_test,
            t_test=t_test,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            weight_decay_lambda=weight_decay,
            name=f"HyperParameter_{i}",
            verbose=False,
        )
        trainer.train()

        # Store the accuracy history
        train_acc_list, val_acc_list = trainer.get_history_accuracy()
        key = f"lr:{lr}, weight decay:{weight_decay}"
        if verbose:
            print(
                f"val acc: {val_acc_list[-1]} | "
                f"lr: {lr}, weight decay: {weight_decay}"
            )
        result[key] = (train_acc_list, val_acc_list)

    return result


def _get_nn_config() -> SequentialConfig:
    # Initialization
    config = SequentialConfig(
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
            AffineConfig(in_size=100, out_size=10, param_suffix="6"),
        ),
    )
    return config
