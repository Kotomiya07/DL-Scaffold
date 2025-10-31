from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """A DataModule for the MNIST dataset."""

    def __init__(self, data_dir: str = "data/", batch_size: int = 64):
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        """Download data if not available."""
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, val, test."""
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.hparams.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.data_test = MNIST(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size)
