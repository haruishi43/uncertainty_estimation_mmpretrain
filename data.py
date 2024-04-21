import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST(
    "./data/mnist",
    download=True,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

data_val = MNIST(
    "./data/mnist",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


def create_dataloader(
    batch_size: int = 1024,
    num_workers: int = 8,
):
    dataloader_train = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=1000,
        num_workers=num_workers,
        drop_last=False,
    )

    dataloaders = {
        "train": dataloader_train,
        "val": dataloader_val,
    }
    return dataloaders


digit_one, _ = data_val[5]
