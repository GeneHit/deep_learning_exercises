import numpy as np
from numpy.typing import NDArray

from common.base import Layer

THRESHOLD_for_MultiLable = 0.9


def calculate_single_label_accuracy(
    network: Layer, x: NDArray[np.floating], t: NDArray[np.floating]
) -> float:
    """
    Calculate the accuracy for a single-label classification model.

    This function evaluates the accuracy of a single-label classification model by
    comparing the predicted class labels with the true labels. The predictions are
    obtained by performing a forward pass through the network and selecting the
    class with the highest score (argmax). It supports both integer class labels
    and one-hot encoded labels for the ground truth.

    Parameters:
        network (Layer): The model (or layer) to evaluate. It must have a `forward`
                         method that accepts the input `x` and returns the predictions.
        x (NDArray[np.floating]): Input data, a NumPy array of shape
                                  `(num_samples, num_features)`.
        t (NDArray[np.floating]): True labels for single-label classification:
                                  - If integer class indices: shape `(num_samples,)`.
                                  - If one-hot encoded: shape `(num_samples, num_classes)`.

    Returns:
        float: The accuracy of the model, calculated as the ratio of correctly
               predicted samples to the total number of samples. The value is
               between 0 and 1.

    Notes:
        - This function is designed for single-label classification tasks where each
          sample belongs to exactly one class.
        - For multi-label classification tasks (where samples may belong to multiple
          classes), use a different function.
    """
    y = network.forward(x)
    y = np.argmax(y, axis=1)  # Get predicted class labels
    if t.ndim != 1:  # Convert one-hot encoded labels to class indices if needed
        t = np.argmax(t, axis=1)

    accuracy = float(np.sum(y == t)) / float(x.shape[0])  # Calculate accuracy
    return accuracy


def calculate_multilabel_accuracy(
    network: Layer, x: NDArray[np.floating], t: NDArray[np.floating]
) -> float:
    """Calculate the accuracy for a multi-label classification model.

    This function evaluates the accuracy of a multi-label classification model by
    comparing the predicted binary labels with the true binary labels. The model's
    output is thresholded to generate predictions, and accuracy is calculated as
    the ratio of samples where all predicted labels exactly match the true labels.

    Parameters:
        network (Layer): The model (or layer) to evaluate. It must have a `forward`
                         method that accepts the input `x` and returns the predictions.
                         The output should be a NumPy array of shape
                         `(num_samples, num_labels)` where each value is the score
                         (e.g., probability) for a label.
        x (NDArray[np.floating]): Input data, a NumPy array of shape
                                  `(num_samples, num_features)`.
        t (NDArray[np.floating]): True binary labels, a NumPy array of shape
                                  `(num_samples, num_labels)` where each value is
                                  either 0 or 1.
        threshold (float, optional): Threshold for converting predicted scores
                                     into binary labels. Default is 0.5.

    Returns:
        float: The multi-label accuracy, calculated as the ratio of samples where
               all predicted labels match the true labels. The value is between 0
               and 1.

    Notes:
        - This function is designed for multi-label classification tasks where each
          sample can belong to multiple labels simultaneously.
        - The accuracy metric requires an exact match of all predicted labels with
          the true labels for each sample. If partial matches are acceptable,
          consider using other metrics like F1-score or Hamming loss.
    """
    # Perform a forward pass through the model to get predictions
    y = network.forward(x)

    # Convert predictions to binary labels using the specified threshold
    y = (y > THRESHOLD_for_MultiLable).astype(np.int8)

    # Check if all predicted labels match the true labels for each sample
    # `all(axis=1)` ensures that every label in a sample must match
    accuracy = float(np.sum((y == t).all(axis=1))) / float(x.shape[0])

    # Return the calculated accuracy
    return accuracy
