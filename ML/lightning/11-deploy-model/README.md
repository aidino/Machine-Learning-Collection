# Deploy your models (advance)

## Deploy with ONNX

https://lightning.ai/docs/pytorch/stable/deploy/production_advanced.html


## Deploy with torchscript

https://lightning.ai/docs/pytorch/stable/deploy/production_advanced_2.html

## Compress models for fast inference

`Pruning` and `Quantization` là các kỹ thuật nén kích thước mô hình để triển khai, cho phép tăng tốc độ suy luận và tiết kiệm năng lượng mà không làm giảm đáng kể độ chính xác.

### Pruning

`Pruning` là một kỹ thuật tập trung vào việc loại bỏ một số trọng số của mô hình để giảm kích thước mô hình và giảm yêu cầu suy luận.

Việc `Pruning` đã được chứng minh là đạt được những cải tiến hiệu quả đáng kể  đồng thời giảm thiểu sự sụt giảm hiệu suất mô hình (chất lượng dự đoán). Nên cắt bớt mô hình cho các cloud endpoints, edge devices hoặc mobile inference 


```python
from lightning.pytorch.callbacks import ModelPruning

# set the amount to be the fraction of parameters to prune
trainer = Trainer(callbacks=[ModelPruning("l1_unstructured", amount=0.5)])
```

You can also perform iterative pruning, apply the lottery ticket hypothesis, and more!

```python
def compute_amount(epoch):
    # the sum of all returned values need to be smaller than 1
    if epoch == 10:
        return 0.5

    elif epoch == 50:
        return 0.25

    elif 75 < epoch < 99:
        return 0.01


# the amount can be also be a callable
trainer = Trainer(callbacks=[ModelPruning("l1_unstructured", amount=compute_amount)])
```