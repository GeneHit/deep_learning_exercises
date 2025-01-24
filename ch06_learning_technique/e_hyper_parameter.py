import numpy as np
from numpy.typing import NDArray


def hyper_parameter_optimization(
    optimization_trial: int,
    x_train: NDArray[np.floating],
    t_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    t_test: NDArray[np.floating],
    weight_decay_bounds: tuple[int, int],
    learning_rate_bounds: tuple[int, int],
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
        weight_decay_bounds (tuple[int, int]): Bounds for weight decay.
        learning_rate_bounds (tuple[int, int]): Bounds for learning rate.
        epochs (int): Number of epochs.
        mini_batch_size (int): Batch size.
        verbose (bool): Verbosity.

    Returns:
        dict[str, tuple[list[float], list[float]]]:
            Dictionary of hyperparameters and train/val accuracy history.
    """
    raise NotImplementedError
