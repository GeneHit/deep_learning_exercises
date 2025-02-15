import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ch08_deep_learning.c_deep_2d_net_pytorch import (
    ModuleTrainer,
    MyDeep2dNet,
    get_device,
    single_label_accuracy,
)


def test_deep_2d_net_pytorch() -> None:
    device = get_device()
    model = MyDeep2dNet().to(device)

    batch_size = 199
    sample_num: int | None = 5000
    target_acc = 0.95
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # get train data (60000, 1, 28, 28)
    train_dataset = datasets.MNIST(
        root="./dataset", train=True, download=True, transform=transform
    )
    if sample_num is not None:
        train_dataset = Subset(train_dataset, range(sample_num))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # get test data (10000, 1, 28, 28)
    test_dataset = datasets.MNIST(
        root="./dataset", train=False, download=True, transform=transform
    )
    if sample_num is not None:
        test_dataset = Subset(test_dataset, range(sample_num))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0
    )

    trainer = ModuleTrainer(
        model=model,
        criterion=criterion,
        evaluation_fn=single_label_accuracy,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=20,
        device=device,
        evaluated_batch_per_epoch=10,
        name="PyTorchNet",
        use_autocast=False,
    )

    trainer.train()

    # Check the final accuracy
    train_acc, test_acc = trainer.get_final_accuracy()
    print(f"train acc, test acc | {train_acc:.4f}, {test_acc:.4f}")
    assert train_acc >= target_acc
    assert test_acc >= target_acc
