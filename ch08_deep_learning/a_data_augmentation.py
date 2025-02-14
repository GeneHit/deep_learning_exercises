import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import rotate, shift

from common.default_type_array import get_default_type


def random_rotation(
    image: NDArray[np.floating], max_angle: int = 15
) -> NDArray[np.floating]:
    """Randomly rotate the image within a specified angle range."""
    angle = np.random.uniform(-max_angle, max_angle)
    rotated_image: NDArray[np.floating] = np.asarray(
        rotate(
            image, angle, axes=(1, 2), reshape=False, mode="constant", cval=0.0
        ),
        dtype=get_default_type(),
    )
    return rotated_image


def random_shift(
    image: NDArray[np.floating], max_shift: int = 2
) -> NDArray[np.floating]:
    """Randomly shift the image within a specified pixel range."""
    assert image.ndim == 3
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    shift_image: NDArray[np.floating] = np.asarray(
        shift(image, [0, shift_x, shift_y], mode="constant", cval=0.0),
        dtype=get_default_type(),
    )
    return shift_image


def random_flip(
    image: NDArray[np.floating], horizontal: bool = True
) -> NDArray[np.floating]:
    """Randomly flip the image horizontally with a 50% probability."""
    if horizontal and np.random.rand() < 0.5:
        return np.fliplr(image)
    return image


def augment_image(image: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply a series of random transformations to an image."""
    image = random_rotation(image)
    image = random_shift(image)
    image = random_flip(image)
    return image


def augment_mnist_data(
    x_train: NDArray[np.floating],
    t_train: NDArray[np.floating],
    augmentation_factor: float = 1.0,
    random_seed: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Perform data augmentation on the MNIST dataset.

    Args:
        x_train (NDArray[np.floating]): Training images, shape (N, C, H, W).
        t_train (NDArray[np.floating]): Training labels, shape (N,).
        augmentation_factor (float): The multiplier for the augmented dataset size.
                                     - augmentation_factor > 1: Increase dataset size.
                                     - augmentation_factor < 1: Reduce dataset size.
        random_seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Tuple[NDArray[np.floating], NDArray[np.floating]]:
            - Augmented training images (NDArray[np.floating]), shape (M, C, H, W).
            - Corresponding labels for the augmented images (NDArray[np.floating]), shape (M,).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # If augmentation_factor is close to 1.0, return the original data
    if np.isclose(augmentation_factor, 1.0):
        return x_train, t_train

    # Move num_original closer to where it's used
    num_original = x_train.shape[0]
    target_size = int(num_original * augmentation_factor)

    # Case 1: Reduce dataset size (augmentation_factor < 1.0)
    if augmentation_factor < 1.0:
        sampled_indices = np.random.choice(num_original, target_size)
        return x_train[sampled_indices], t_train[sampled_indices]

    # Case 2: Increase dataset size (augmentation_factor > 1.0)
    # idea: use a np.random.choice to generate the indices

    sampled_indices = np.random.choice(
        num_original, (target_size - num_original)
    )
    for idx in sampled_indices:
        augmented_image = augment_image(x_train[idx])
        x_train = np.append(x_train, [augmented_image], axis=0)
        t_train = np.append(t_train, [t_train[idx]], axis=0)

    return x_train, t_train
