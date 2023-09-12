# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class FakeDataset1(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset1", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


class FakeDataset2(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset2", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


cli = LightningCLI(DemoModel)


# use Model1
# python multi-lightning-datamodules.py fit --data FakeDataset1

# use Model2
# python multi-lightning-datamodules.py fit --data FakeDataset2