import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ch06_learning_technique.e_hyper_parameter import (
    hyper_parameter_optimization,
)
from dataset.mnist import load_mnist


def shuffle_dataset(
    x: NDArray[np.floating], t: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    permutation = np.random.permutation(x.shape[0])
    return x[permutation], t[permutation]


def test_hyper_parameter_optimization() -> None:
    # Load MNIST data, returning a 60000x784 array for x and a nx10 array for t
    ((x_train, t_train), _) = load_mnist(normalize=True)

    x_train = x_train[:500]
    t_train = t_train[:500]

    # shuffle the dataset avoiding the same order
    x_train, t_train = shuffle_dataset(x_train, t_train)

    # split the dataset into training and validation sets
    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)
    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]

    results = hyper_parameter_optimization(
        optimization_trial=100,
        x_train=x_train,
        t_train=t_train,
        x_test=x_val,
        t_test=t_val,
        weight_decay_log_bounds=(-8, -4),
        learning_rate_log_bounds=(-6, -2),
        epochs=50,
        mini_batch_size=100,
        verbose=False,
    )

    assert len(results) == 100

    # set to True if you want to see the result
    show_result = False
    if show_result:
        _plot_result(results)


def _plot_result(results: dict[str, tuple[list[float], list[float]]]) -> None:
    print("=========== Hyper-Parameter Optimization Result ===========")
    graph_draw_num = 20
    col_num = 5
    row_num = int(np.ceil(graph_draw_num / col_num))
    i = 0

    # Sort results by last validation accuracy
    sorted_results = sorted(
        results.items(), key=lambda x: x[1][1][-1], reverse=True
    )

    _, axes = plt.subplots(row_num, col_num, figsize=(15, 8))
    axes = axes.flatten()  # Convert to a flat list for easier indexing

    for key, acc_list in sorted_results:
        val_acc_list = acc_list[1]
        print(f"Best-{i + 1} (val acc: {val_acc_list[-1]}) | {key}")

        ax = axes[i]
        ax.set_title(f"Best-{i + 1}")
        ax.set_ylim(0.0, 1.0)

        if i % col_num:
            ax.set_yticks([])
        ax.set_xticks([])

        x = np.arange(len(val_acc_list))
        # Validation accuracy with solid line
        ax.plot(x, val_acc_list, label="Validation Accuracy", linestyle="-")
        # Training accuracy with dashed line
        ax.plot(x, acc_list[0], label="Training Accuracy", linestyle="--")

        i += 1
        if i >= graph_draw_num:
            break

    # Add legend to the last subplot
    if i > 0:
        axes[i - 1].legend(loc="upper left")  # Legend only in the last subplot

    plt.tight_layout()
    plt.show()
