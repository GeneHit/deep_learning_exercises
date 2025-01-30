import numpy as np


def sigmoid(
    x: np.typing.NDArray[np.floating],
) -> np.typing.NDArray[np.floating]:
    """Sigmoid activation function.

    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))

    Parameters:
        x (np.ndarray of float): Input array.

    Returns:
        np.ndarray of float: Output array after applying sigmoid function.
    """
    result: np.typing.NDArray[np.floating] = 1 / (1 + np.exp(-x)) # for mypy
    return result


def step(x: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
    """Step activation function.

    The step function is defined as:
        step(x) = 1 if x > 0 else 0

    Parameters:
        x (np.ndarray of float): Input array.

    Returns:
        np.ndarray of int: Output array after applying step function.
    """
    return np.where(x > 0, 1, 0)


def relu(x: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
    """ReLU activation function.

    The ReLU function is defined as:
        relu(x) = x if x > 0 else 0

    Parameters:
        x (np.ndarray of float): Input array.

    Returns:
        np.ndarray of float: Output array after applying ReLU function.
    """
    return np.maximum(0, x)


def identity_function(
    x: np.typing.NDArray[np.floating],
) -> np.typing.NDArray[np.floating]:
    """Identity function.

    The identity function is defined as:
        identity_function(x) = x

    Parameters:
        x (np.ndarray of float): Input array.

    Returns:
        np.ndarray of float: Output array after applying identity function.
    """
    return x


def softmax(
    x: np.typing.NDArray[np.floating],
) -> np.typing.NDArray[np.floating]:
    """
    Softmax activation function.

    The softmax function is defined as:
        softmax(x) = exp(x) / sum(exp(x))

    The sum is calculated over all elements in the input array, ensuring that the output
    values are normalized and sum to 1. This makes the softmax function particularly useful
    for multi-class classification problems.

    Parameters:
        x (np.ndarray of float): Input array.

    Returns:
        np.ndarray of float: Output array after applying softmax function.
    """
    # Subtracting the maximum value ensures numerical stability.
    exp_x = np.exp(x - np.max(x))
    result: np.typing.NDArray[np.floating] = exp_x / np.sum(exp_x)  # for mypy
    return result
