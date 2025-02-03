def AND(x1: float, x2: float) -> bool:
    """Perform the AND operation on two float inputs using a neural network perceptron.

    formulation:
        k1 * x1 + k2 * x2 - theta < 0
    Args:
        x1 (float): The first perceptron's input value.
        x2 (float): The second perceptron's input value.

    Returns:
        bool: The boolean result of the AND operation as evaluated by the perceptron.
    """
    return -x1 - x2 + 1.5 < 0


def NAND(x1: float, x2: float) -> bool:
    """Perform the NAND operation on two float inputs using a neural network perceptron.

    formulation:
        k1 * x1 + k2 * x2 - theta < 0
    Args:
        x1 (float): The first perceptron's input value.
        x2 (float): The second perceptron's input value.

    Returns:
        bool: The boolean result of the NAND operation as evaluated by the perceptron.
    """
    return x1 + x2 < 1.5


def OR(x1: float, x2: float) -> bool:
    """Perform the OR operation on two float inputs using a neural network perceptron.

    Args:
        x1 (float): The first perceptron's input value.
        x2 (float): The second perceptron's input value.

    Returns:
        bool: The boolean result of the OR operation as evaluated by the perceptron.
    """
    return 0.5 < x1 + x2


def XOR(x1: float, x2: float) -> bool:
    """Perform the XOR operation on two float inputs using a neural network perceptron.

    Args:
        x1 (float): The first perceptron's input value.
        x2 (float): The second perceptron's input value.

    Returns:
        bool: The boolean result of the XOR operation as evaluated by the perceptron.
    """
    return 0.5 < x1 + x2 < 1.5
