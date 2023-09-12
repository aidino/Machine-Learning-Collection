# Lightining Data Module

## Lightning Data Module APIs

### `prepare_data()`

- Downloading and saving data with multiple workers (distributed setting)
- Run on CPU only
- In case of multiple nodes, the execution of this hook depends upon `prepare_data_per_node()`
- `setup()` is called after `prepare_data()`
  
```python
class MNISTDataModule(pl.LightningDataModule):
    def prepare_data(self):
        # download
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
```

### `setup()`

There are also data operations you might want to perform on every `GPU`. Use `setup()` to do things like:

- count number of classes
- build vocabulary
- perform train/val/test splits
- create datasets
- apply transforms (defined explicitly in your datamodule)
- etc…

```python
import lightning.pytorch as pl


class MNISTDataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
```

```python
class LitDataModule(LightningDataModule):
    def prepare_data(self):
        dataset = load_Dataset(...)
        train_dataset = ...
        val_dataset = ...
        # tokenize
        # save it to disk

    def setup(self, stage):
        # load it back here
        dataset = load_dataset_from_disk(...)
```

### `transfer_batch_to_device`

Sử dụng hàm này nếu dữ liệu trả về từ `Dataloader` được gói trong các kiểu dữ liệu tùy chỉnh ngoài các kiểu dữ liệu sau:

- `torch.Tensor` r anything that implements `.to(…)`
- `list`
- `dict`
- `tuple`

For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, …).

```python
def transfer_batch_to_device(self, batch, device, dataloader_idx):
    if isinstance(batch, CustomBatch):
        # move all tensors in your custom data structure to the device
        batch.samples = batch.samples.to(device)
        batch.targets = batch.targets.to(device)
    elif dataloader_idx == 0:
        # skip device transfer for the first dataloader or anything you wish
        pass
    else:
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
    return batch
```
...
[More infomation](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)


### Using `DataModule`

```python
dm = MNISTDataModule()
model = Model()
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
trainer.validate(datamodule=dm)
trainer.predict(datamodule=dm)
```

### `DataModules` without Lightning

```python
# download, etc...
dm = MNISTDataModule()
dm.prepare_data()

# splits/transforms
dm.setup(stage="fit")

# use data
for batch in dm.train_dataloader():
    ...

for batch in dm.val_dataloader():
    ...

dm.teardown(stage="fit")

# lazy load test data
dm.setup(stage="test")
for batch in dm.test_dataloader():
    ...

dm.teardown(stage="test")
```

### Hyperparameters in `DataModules`

Like LightningModules, DataModules support hyperparameters with the same API.

```python
import lightning.pytorch as pl


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        # access the saved hyperparameters
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr)
```

### Save DataModule state

```python
class LitDataModule(pl.DataModuler):
    def state_dict(self):
        # track whatever you want here
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.current_train_batch_index = state_dict["current_train_batch_index"]
```