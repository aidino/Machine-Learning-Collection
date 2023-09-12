# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class LitLRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
    def step(self):
        print("⚡", "using LitLRScheduler", "⚡")
        super().step()


cli = LightningCLI(DemoModel, BoringDataModule)

# LitLRScheduler
# python main.py fit --lr_scheduler LitLRScheduler