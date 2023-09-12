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
    
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=20)
    valid_dataloader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=20)
    
    # 01:37, 490.73it/s
    # trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1)
    
    # 02:02, 391.65it/s
    # added precision='16-mixed'
    # trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1, precision='16-mixed')
    
    
    # 00:06, 232.90it/s
    # added num_workers=20 to data loader
    # trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1)
    
    # 00:06, 232.90it/s
    # added accumulate_grad_batches=16 to data loader
    # trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=1, accumulate_grad_batches=16)
    
    # 00:07, 207.03it/s
    # trainer = pl.Trainer(accelerator="gpu", devices=1, strategy="ddp", num_nodes=1, max_epochs=1)
    
    # 00:07<00:00, 203.65it/s
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, max_epochs=1)
    
    trainer.fit(
        LitAutoEncoder(), 
        train_dataloader, 
        valid_dataloader)
    

if __name__ == '__main__':
    main()
    