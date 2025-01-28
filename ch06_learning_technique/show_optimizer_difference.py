import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ch06_learning_technique.a_optimization import (
    SGD,
    AdaGrad,
    Adam,
    Momentum,
)
from common.base import Optimizer


def f(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
    return x**2 / 20.0 + y**2


def df(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    return x / 10.0, 2 * y


def process_one_optimizer(
    optimizer: Optimizer,
    init_pos: tuple[float, float],
    step: int,
) -> tuple[list[float], list[float]]:
    """Process the data of a optimizer.

    Parameters:
        optimizer : (Optimizer)
            Optimizer.
        init_pos (tuple[float, float]): Initial position.
        step (int): Number of steps.

    Returns:
        tuple[list[float], list[float]]: x and y history.
    """
    x_history = []
    y_history = []

    grads = {}
    pos: dict[str, NDArray[np.floating]] = {
        "x": np.array([init_pos[0]]),
        "y": np.array([init_pos[1]]),
    }
    for _ in range(step):
        x_history.append(pos["x"][0])
        y_history.append(pos["y"][0])

        grads["x"], grads["y"] = df(pos["x"], pos["y"])
        optimizer.one_step(pos, grads)

    return x_history, y_history


def show_optimizer_difference(
    optimizers: dict[str, Optimizer],
    init_pos: tuple[float, float],
    step: int,
    grid_x_range: tuple[float, float, float] = (-10, 10, 0.01),
    grid_y_range: tuple[float, float, float] = (-5, 5, 0.01),
) -> None:
    """Show the difference between the optimizers.

    Parameters:
        optimizers : (dict[str, Optimizer])
            Dictionary of optimizers, with the name as the key.
        init_pos (tuple[float, float]): Initial position.
        step (int): Number of steps.
        grid_x_range : (tuple[float, float, float])
            Range of x values, with the start, end, and step size.
        grid_y_range : (tuple[float, float, float])
            Range of y values, with the start, end, and step size.
    """
    subplot_size = (2, (len(optimizers) + 1) // 2)
    subplot_idx = 1
    for key, optimizer in optimizers.items():
        x_history, y_history = process_one_optimizer(optimizer, init_pos, step)

        X, Y = np.meshgrid(np.arange(*grid_x_range), np.arange(*grid_y_range))
        Z = f(X, Y)
        # for simple contour line
        mask = Z > 7
        Z[mask] = 0

        # plot
        plt.subplot(subplot_size[0], subplot_size[1], subplot_idx)
        subplot_idx += 1
        plt.plot(x_history, y_history, "o-", color="red")
        plt.contour(X, Y, Z)
        plt.ylim(grid_x_range[0], grid_x_range[1])
        plt.xlim(grid_y_range[0], grid_y_range[1])
        plt.plot(0, 0, "+")
        # colorbar()
        # spring()
        plt.title(key)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.show()


def compare_optimizers() -> None:
    optimizers: dict[str, Optimizer] = {
        "SGD": SGD(lr=0.95),
        "Momentum": Momentum(lr=0.1),
        "AdaGrad": AdaGrad(lr=1.5),
        "Adam": Adam(lr=0.3),
    }
    show_optimizer_difference(optimizers, init_pos=(-7.0, 2.0), step=30)


if __name__ == "__main__":
    compare_optimizers()
