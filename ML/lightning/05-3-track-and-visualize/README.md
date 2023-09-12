# TRACK AND VISUALIZE EXPERIMENTS (BASIC)

## Track metrics

```python
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        value = ...
        self.log("some_value", value)

# To log multiple metrics at once, use self.log_dict

values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
self.log_dict(values)


```