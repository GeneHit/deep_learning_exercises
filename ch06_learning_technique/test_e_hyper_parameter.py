import matplotlib.pyplot as plt
import numpy as np

from ch06_learning_technique.e_hyper_parameter import (
    hyper_parameter_optimization,
)
from common.utils import shuffle_dataset
from dataset.mnist import load_mnist


def test_hyper_parameter_optimization() -> None:
    # Load MNIST data, returning a 60000x784 array for x and a nx10 array for t
    ((x_train, t_train), _) = load_mnist(normalize=True)

    x_train = x_train[:500]
    t_train = t_train[:500]
    validation_rate = 0.20
    validation_num = int(x_train.shape[0] * validation_rate)
    x_train, t_train = shuffle_dataset(x_train, t_train)
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
        weight_decay_bounds=(-8, -4),
        learning_rate_bounds=(-6, -2),
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

    for key, acc_list in sorted(
        results.items(), key=lambda x: x[1][1][-1], reverse=True
    ):
        val_acc_list = acc_list[1]
        print(
            "Best-"
            + str(i + 1)
            + "(val acc:"
            + str(val_acc_list[-1])
            + ") | "
            + key
        )

        plt.subplot(row_num, col_num, i + 1)
        plt.title("Best-" + str(i + 1))
        plt.ylim(0.0, 1.0)
        if i % 5:
            plt.yticks([])
        plt.xticks([])
        x = np.arange(len(val_acc_list))
        plt.plot(x, val_acc_list)
        plt.plot(x, acc_list[0], "--")
        i += 1

        if i >= graph_draw_num:
            break

    plt.show()
