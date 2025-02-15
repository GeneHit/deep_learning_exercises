from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MyDeep2dNet(nn.Module):
    """A deep 2D convolutional neural network for mnist dataset.

    Structural diagram:
        input: (1, 28, 28) with mnist dataset
        Layer 1: Convolutional Layer
            - Filters: 16
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (16, 28, 28)

        Layer 2: Convolutional Layer
            - Filters: 16
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (16, 28, 28)

        Layer 3: Max Pooling Layer
            - Pool size: (2, 2)
            - Stride: 2
            - Output: (16, 14, 14)

        Layer 4: Convolutional Layer
            - Filters: 32
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (32, 14, 14)

        Layer 5: Convolutional Layer
            - Filters: 32
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (32, 14, 14)

        Layer 6: Max Pooling Layer
            - Pool size: (2, 2)
            - Stride: 2
            - Output: (32, 7, 7)

        Layer 7: Convolutional Layer
            - Filters: 64
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (64, 7, 7)

        Layer 8: Convolutional Layer
            - Filters: 64
            - Filter size: (3, 3)
            - Stride: 1
            - Padding: 1
            - Activation: ReLU
            - Output: (64, 7, 7)

        Layer 9: Max Pooling Layer
            - Pool size: (2, 2)
            - Stride: 2
            - Output: (64, 3, 3)

        Layer 10: Fully Connected Layer
            - Neurons: 50
            - Activation: ReLU
            - Output: (50,)

        Layer 11: Dropout Layer
            - Dropout ratio: 0.5

        Layer 11: Fully Connected Layer
            - Neurons: 10
            - Activation:
            - Output: (10,)

        layer 12: Dropout Layer
            - Dropout ratio: 0.5
    """

    def __init__(self) -> None:
        super(MyDeep2dNet, self).__init__()
        # use the nn.Sequential model for the network
        self.model = nn.Sequential(
            # Layer 1: (N, 1, 28, 28) -> (N, 16, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # Layer 2: (N, 16, 28, 28) -> (N, 16, 28, 28)
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # Layer 3: (N, 16, 28, 28) -> (N, 16, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 4: (N, 16, 14, 14) -> (N, 32, 14, 14)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # Layer 5: (N, 32, 14, 14) -> (N, 32, 14, 14)
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # Layer 6: (N, 32, 14, 14) -> (N, 32, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 7: (N, 32, 7, 7) -> (N, 64, 7, 7)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # Layer 8: (N, 64, 7, 7) -> (N, 64, 7, 7)
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # Layer 9: (N, 64, 7, 7) -> (N, 64, 3, 3)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 10: (N, 64, 3, 3) -> (N, 50)
            nn.Flatten(),
            nn.Linear(in_features=64 * 3 * 3, out_features=50),
            nn.ReLU(),
            # Layer 11: Dropout
            nn.Dropout(p=0.5),
            # Layer 12: (N, 50) -> (N, 10)
            nn.Linear(in_features=50, out_features=10),
            nn.Dropout(p=0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the network.

        Parameters:
            x : (Tensor)
                The input data to the network, a PyTorch tensor of shape
                `(batch_size, 1, 28, 28)`.

        Returns:
            (Tensor): The output of the network, a PyTorch tensor of shape
                      `(batch_size, 10)`.
        """
        y: Tensor = self.model(x)  # y for mypy
        return y


class ModuleTrainer:
    """A trainer for training a neural network.

    This is for the network that is based on the pytorch.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        evaluation_fn: Callable[[Tensor, Tensor], float],
        optimizer: optim.Optimizer,
        train_loader: DataLoader[TensorDataset],
        test_loader: DataLoader[TensorDataset],
        epochs: int,
        device: torch.device,
        evaluate_train_data: bool = True,
        evaluate_test_data: bool = True,
        evaluated_batch_per_epoch: int | None = None,
        verbose: bool = False,
        name: str = "",
        use_autocast: bool = False,
    ) -> None:
        """Initialize the trainer.

        Parameters:
            model : (nn.Module)
                The neural network model to train.
            criterion : (nn.Module)
                The loss function to optimize.
            evaluation_fn : (Callable[[Tensor, Tensor], float])
                A function to evaluate the model's performance.
            optimizer : (optim.Optimizer)
                The optimization algorithm to use.
            train_loader : (DataLoader)
                DataLoader for the training dataset.
            test_loader : (DataLoader)
                DataLoader for the test dataset.
            epochs : (int)
                The number of epochs to train the model.
            device : (torch.device)
                The device to run the model on (CPU or GPU).
            evaluate_train_data : (bool, optional)
                Whether to evaluate the model on the training data. Default is True.
            evaluate_test_data : (bool, optional)
                Whether to evaluate the model on the test data. Default is True.
            evaluated_batch_per_epoch : (int, optional)
                The number of batches to evaluate per epoch. Default is None.
            verbose : (bool, optional)
                Whether to output detailed logs during training. Default is False.
            name : (str, optional)
                The name of the training process. Default is an empty string.
            use_autocast : bool
                Whether to use automatic mixed precision training. Default is False.
                It is only for cuda GPU devices.
        """
        self._model = model
        self._criterion = criterion
        self._evaluation_fn = evaluation_fn
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._epochs = epochs
        self._device = device
        self._evaluate_train_data = evaluate_train_data
        self._evaluate_test_data = evaluate_test_data
        self._evaluated_batch_per_epoch = evaluated_batch_per_epoch
        self._verbose = verbose
        self._name = name
        self._use_autocast = use_autocast

        self._reset_history()

    def _reset_history(self) -> None:
        self.runnig_losses: list[float] = []
        self.train_acc_history: list[float] = []
        self.test_acc_history: list[float] = []
        self._final_accuracy: tuple[float, float] | None = None

    def train(self) -> None:
        self._reset_history()

        # tqdm progress bar for epochs
        desc = self._name if self._name else "Training Progress"
        epoch_bar = tqdm(range(self._epochs), desc=desc)
        for epoch in epoch_bar:
            if self._use_autocast:
                self._train_one_epoch_with_autocast()
            else:
                self._train_one_epoch()

            # output the necessary logging if necessary
            self._evaluate_if_necessary(epoch)

    def get_final_accuracy(self) -> tuple[float, float]:
        """Get the final accuracy of the network after training.

        Have to call the train method before calling this method.

        Returns:
            tuple[float, float]: Train and test accuracy.
        """
        if self._final_accuracy is not None:
            return self._final_accuracy

        self._final_accuracy = (
            self._evaluate(data_loader=self._train_loader, process="Train Acc"),
            self._evaluate(data_loader=self._test_loader, process="Test Acc"),
        )
        return self._final_accuracy

    def get_history_accuracy(self) -> tuple[list[float], list[float]]:
        """Get the history of the training and test accuracy."""
        return self.train_acc_history, self.test_acc_history

    def _evaluate_if_necessary(self, epoch: int) -> None:
        if not self._evaluate_train_data and not self._evaluate_test_data:
            return

        # output the training accuracy if necessary
        if self._evaluate_train_data:
            train_acc = self._evaluate(
                data_loader=self._train_loader,
                process=f"Epoch {epoch + 1} Training",
                num_batch=self._evaluated_batch_per_epoch,
            )
            self.train_acc_history.append(train_acc)

        # output the test accuracy if necessary
        if self._evaluate_test_data:
            test_acc = self._evaluate(
                data_loader=self._test_loader,
                process=f"Epoch {epoch + 1} Test",
                num_batch=self._evaluated_batch_per_epoch,
            )
            self.test_acc_history.append(test_acc)

    def _evaluate(
        self,
        data_loader: DataLoader[TensorDataset],
        process: str,
        num_batch: int | None = None,
    ) -> float:
        batch_num = len(data_loader)
        if num_batch is not None:
            batch_num = min(batch_num, num_batch)

        acc_sum = 0.0
        loss_sum = 0.0
        self._model.eval()
        with torch.no_grad():
            for i, (x, t) in enumerate(data_loader):
                if i >= batch_num:
                    break
                x, t = x.to(self._device), t.to(self._device)
                y: Tensor = self._model(x)
                acc_sum += self._evaluation_fn(y, t)
                if self._verbose:
                    loss: Tensor = self._criterion(y, t)
                    loss_sum += loss.item()

        acc = acc_sum / batch_num
        if self._verbose:
            print(f"{process}: Acc {acc:.4f}; Loss {loss_sum / batch_num:.4f}")
        return acc

    def _train_one_epoch(self) -> None:
        """Train the network for one epoch.

        Steps every iteration:
            - Get the mini-batch in device
            - zero grad and forward
            - Calculate the loss
            - Backward
            - Update the parameters once
        """
        self._model.train()
        running_loss = 0.0
        for x, t in self._train_loader:
            x, t = x.to(self._device), t.to(self._device)
            self._optimizer.zero_grad()
            y = self._model(x)
            loss: Tensor = self._criterion(y, t)
            running_loss += loss.item()
            loss.backward()
            self._optimizer.step()
        self.runnig_losses.append(running_loss / len(self._train_loader))

    def _train_one_epoch_with_autocast(self) -> None:
        """Train the network for one epoch with mix precision training."""
        self._model.train()
        running_loss = 0.0
        scaler = GradScaler()
        for x, t in self._train_loader:
            x, t = x.to(self._device), t.to(self._device)
            self._optimizer.zero_grad()
            with autocast():
                y = self._model(x)
                loss: Tensor = self._criterion(y, t)
            running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(self._optimizer)
            scaler.update()
        self.runnig_losses.append(running_loss / len(self._train_loader))


def single_label_accuracy(model_output: Tensor, target: Tensor) -> float:
    """Calculate the accuracy for a single-label classification model.

    This function evaluates the accuracy of a single-label classification model by
    comparing the predicted class labels with the true labels. The predictions are
    obtained by performing a forward pass through the network and selecting the
    class with the highest score (argmax). It supports both integer class labels
    and one-hot encoded labels for the ground truth.

    Parameters:
        model_output : (Tensor)
            Output of the model, a PyTorch tensor of shape
            `(num_samples, num_classes)`.
        target : (Tensor)
            True labels for single-label classification:
                - If integer class indices: shape `(num_samples,)`.
                - If one-hot encoded: shape `(num_samples, num_classes)`.

    Returns:
        float: The accuracy of the model, calculated as the ratio of correctly
               predicted samples to the total number of samples. The value is
               between 0 and 1.
    """
    assert model_output.ndim == 2, (
        "Model output must have shape (num_samples, num_classes)"
    )
    assert target.ndim in {1, 2}, (
        "Target must have shape (num_samples,) or (num_samples, num_classes)"
    )
    if target.ndim == 2:
        target = target.argmax(dim=1)
    return (model_output.argmax(dim=1) == target).float().mean().item()


def get_device() -> torch.device:
    """Get the device to run the model on (CPU, GPU, or MPS).

    Returns:
        torch.device: The device to run the model on.
    """
    device = "cpu"
    if torch.cuda.is_available():
        # GPU is available
        device = "cuda"
    elif torch.backends.mps.is_available():
        # Multi-Process Service (MPS) of Apple chip (M1/M2) is available
        device = "mps"
    return torch.device(device)
