import numpy as np
from numpy.typing import NDArray

WEIGHT_START_WITH = "W"


def weight_square_sum(params: dict[str, NDArray[np.floating]]) -> float:
    weight_decay = 0.0
    for key in params.keys():
        if key.startswith(WEIGHT_START_WITH):
            weight_decay += np.sum(params[key] ** 2)

    return weight_decay


def update_weight_decay_if_necessary(
    params_grad: dict[str, NDArray[np.floating]],
    params: dict[str, NDArray[np.floating]],
    weight_decay_lambda: float | None,
) -> None:
    if weight_decay_lambda:
        for key in params_grad.keys():
            if key.startswith(WEIGHT_START_WITH):
                params_grad[key] += weight_decay_lambda * params[key]
