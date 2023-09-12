# DEPLOY MODELS INTO PRODUCTION (BASIC)

## Load a checkpoint and predict

```python
model = LitModel.load_from_checkpoint("best_model.ckpt")
model.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)
```

## Predict step with your LightningModule

```python
class MyModel(LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

# And pass in any dataloader to the Lightning Trainer:

data_loader = DataLoader(...)
model = MyModel()
trainer = Trainer()
predictions = trainer.predict(model, data_loader)
```

## Enable complicated predict logic

When you need to add complicated pre-processing or post-processing logic to your data use the predict step. For example here we do Monte Carlo Dropout for predictions:

```python
class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iteration):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iteration

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        return pred
```

## Enable distributed inference

By using the predict step in Lightning you get free distributed inference using `BasePredictionWriter`.

```python
import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


# or you can set `write_interval="batch"` and override `write_on_batch_end` to save
# predictions at batch level
pred_writer = CustomWriter(output_dir="pred_path", write_interval="epoch")
trainer = Trainer(accelerator="gpu", strategy="ddp", devices=8, callbacks=[pred_writer])
model = BoringModel()
trainer.predict(model, return_predictions=False)
```