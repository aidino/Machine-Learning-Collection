import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from mnist_module import LitAutoEncoder

def main():
    # define the training dataset
    dataset = MNIST('../data/', download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)
    
    # Train the model using Trainer
    autoencoder = LitAutoEncoder()
    trainer = pl.Trainer()
    trainer.fit(autoencoder, train_loader)
    
    
    # Train the model under the hood
    # autoencoder = LitAutoEncoder()
    # optimizer = autoencoder.configure_optimizers()
    # for batch_idx, batch in enumerate(train_loader):
    #     loss = autoencoder.training_step(batch, batch_idx)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    

if __name__ == '__main__':
    main()
    
# Source: https://lightning.ai/docs/pytorch/latest/model/train_model_basic.html