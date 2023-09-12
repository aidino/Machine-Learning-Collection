import torch
import lightning.pytorch as pl
import torchvision

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '../data'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # (how to download, tokenize, etc…)
    # CPU only
    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)
    
    # (how to split, define dataset, etc…)
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit':
            mnist_full = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [55000, 5000])
        
        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        
        if stage == 'predict':
            self.mnist_predict = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=64)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=64)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=64)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_predict, batch_size=64)
    