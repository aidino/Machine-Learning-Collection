from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demo.boring_classes import DemoModel, BoringDataModule

class Model1(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()


cli = LightningCLI(datamodule_class=BoringDataModule)

# use Model1
# python multi-lightning-modules.py fit --model Model1

# use Model2
# python multi-lightning-modules.py fit --model Model2