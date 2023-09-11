import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import ModelSummary
from mnist_module import LitAutoEncoder
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor

def main():
    # define the training dataset
    transform = transforms.ToTensor()
    train_set = MNIST('../data/', download=True, train=True, transform=transform)
    test_set = MNIST('../data/', download=True, train=False, transform=transform)
    
    # Use 20% of training data as validation data
    seed = torch.Generator().manual_seed(42)
    train_set_size = int(len(train_set) * 0.8)
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = data.random_split(train_set, [train_set_size, val_set_size], generator=seed)
    
    train_dataloader = DataLoader(train_set)
    valid_dataloader = DataLoader(val_set)
    
    model = LitAutoEncoder()
    
    summary = ModelSummary(model, max_depth=-1)
    
    print(summary)
    
    # Simple profiler
    # trainer = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.01, max_epochs=3, profiler="simple")
    
    # Advanced profiler to file
    # profiler = AdvancedProfiler(dirpath="profiler_logs", filename="profiler_log.txt")
    # trainer = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.01, max_epochs=3, profiler=profiler)
    
    # Measure accelerator usage
    trainer = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.01, max_epochs=3, callbacks=[DeviceStatsMonitor()])
    
    trainer.fit(
        model, 
        train_dataloader,  
        valid_dataloader)
    

if __name__ == '__main__':
    main()
    
# Source: https://lightning.ai/docs/pytorch/latest/common/checkpointing_basic.html