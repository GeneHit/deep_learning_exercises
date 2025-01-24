import numpy as np
from numpy.typing import NDArray


def shuffle_dataset(
    x: NDArray[np.floating], t: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    permutation = np.random.permutation(x.shape[0])
    return x[permutation], t[permutation]
