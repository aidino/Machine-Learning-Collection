import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from mnist_module import LitAutoEncoder

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
    
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=3)
    trainer.fit(LitAutoEncoder(), train_dataloader, valid_dataloader)
    

if __name__ == '__main__':
    main()
    
# Source: https://lightning.ai/docs/pytorch/latest/common/evaluation_basic.html