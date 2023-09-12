# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class LitAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using LitAdam", "⚡")
        super().step(closure)


class FancyAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using FancyAdam", "⚡")
        super().step(closure)


cli = LightningCLI(DemoModel, BoringDataModule)

# use LitAdam
# python main.py fit --optimizer LitAdam

# use FancyAdam
# python main.py fit --optimizer FancyAdam