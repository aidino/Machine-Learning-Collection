# Explore SOTA scaling techniques

## N-bit precision

[Users looking to train models faster and consume less memory.]

Trong quá trình huấn luyện mạng nơ-ron trong PyTorch, `"N-BIT PRECISION"` thường liên quan đến cách chúng ta biểu diễn và xử lý dữ liệu số trong mạng nơ-ron. `N-BIT PRECISION` đề cập đến số lượng bit được sử dụng để biểu diễn giá trị số học, ví dụ: số thực hoặc số nguyên, trong mạng nơ-ron.

Khi nói về  `N-BIT PRECISION`, ta đang nói về cách mà mạng nơ-ron sử dụng bộ dữ liệu số học với một số lượng bit nhất định để lưu trữ và tính toán. Điều này có thể ảnh hưởng đến độ chính xác và tốc độ của quá trình huấn luyện.

Ví dụ, một mạng nơ-ron được thiết lập để sử dụng `16-BIT PRECISION` sẽ sử dụng 16 bit để lưu trữ các giá trị số học, **giảm đi độ chính xác** so với mạng sử dụng 32-bit (sử dụng cho số thực độ chính xác đơn). Điều này có thể làm giảm khả năng biểu diễn chính xác của mạng, nhưng sẽ tiết kiệm bộ nhớ và tăng tốc độ tính toán.

Trong một số tình huống, việc sử dụng `N-BIT PRECISION` có thể được coi là một phần của kỹ thuật tối ưu hóa để đạt được sự cân bằng giữa độ chính xác và hiệu suất trong quá trình huấn luyện của mạng nơ-ron. Tùy thuộc vào ứng dụng cụ thể, bạn có thể điều chỉnh `N-BIT PRECISION` để đáp ứng yêu cầu của mình và tối ưu hóa quá trình huấn luyện.

### `16-bit Precision`

```python
Trainer(precision='16-mixed')
```

### `32-bit Precision`

This is the default used across all models and research.

```python
Trainer(precision="32-true")

# or
Trainer(precision="32")

# or
Trainer(precision=32)
```

### `64-bit Precision`
- Trong một số tính toán khoa học, 64-bit precision dùng để nâng cao độ chính xác. Tuy nhiên, Double 32-bit = 64 bit đồng nghĩa với việc double the memory requirements.

```python
Trainer(precision="64-true")

# or
Trainer(precision="64")

# or
Trainer(precision=64)
```

| **Precision**  | **CPU** | **GPU** | **TPU** | **IPU** |
|----------------|---------|---------|---------|---------|
| 16 Mixed       | No      | Yes     | No      | Yes     |
| BFloat16 Mixed | Yes     | Yes     | Yes     | No      |
| 32 True        | Yes     | Yes     | Yes     | Yes     |
| 64 True        | Yes     | Yes     | No      | No      |


## SOTA scaling techniques

Lightning implements various techniques to help during training that can help make the training smoother.

### Accumulate Gradients

Phương pháp "`Accumulate Gradients`" (tạm dịch: tích lũy gradient) là một kỹ thuật trong quá trình huấn luyện mạng nơ-ron để xử lý vấn đề về bộ nhớ và tối ưu hiệu suất. 

Trong quá trình huấn luyện mạng nơ-ron, ta thường sử dụng một phương pháp gọi là "Backpropagation" để tính toán gradient của hàm mất mát đối với các tham số của mạng. Gradient này sau đó được sử dụng để cập nhật các tham số này trong quá trình tối ưu hóa mô hình (ví dụ: sử dụng thuật toán gradient descent).

Khi làm việc với các bộ dữ liệu lớn hoặc mạng nơ-ron cồng kềnh, việc tính toán gradient có thể tiêu tốn nhiều bộ nhớ. Điều này có thể dẫn đến việc cần có bộ nhớ GPU lớn hơn để xử lý huấn luyện hoặc thậm chí không đủ bộ nhớ để huấn luyện mô hình.

Kỹ thuật "Accumulate Gradients" giải quyết vấn đề này bằng cách thay vì cập nhật tham số mỗi lần sau khi tính toán gradient cho từng mini-batch (tập con của dữ liệu đang huấn luyện), ta tích lũy gradient từ nhiều mini-batch lại. Sau khi tích lũy đủ gradient từ một số mini-batch, ta mới cập nhật tham số của mạng. Việc này giúp giảm bộ nhớ cần thiết cho quá trình huấn luyện, vì ta không cần lưu trữ gradient của từng mini-batch riêng lẻ.

Kỹ thuật tích lũy gradient có thể được điều chỉnh thông qua một siêu tham số gọi là "accumulation steps" (số bước tích lũy), cho phép bạn kiểm soát tần suất cập nhật tham số. Tùy thuộc vào kích thước bộ nhớ và yêu cầu cụ thể của bài toán, bạn có thể điều chỉnh giá trị này để tối ưu hóa hiệu suất huấn luyện của mạng nơ-ron.

```python
# DEFAULT (ie: no accumulated grads)
trainer = Trainer(accumulate_grad_batches=1)

# Accumulate gradients for 7 batches
trainer = Trainer(accumulate_grad_batches=7)
```

Chúng ta có thể thay đổi `accumulate_grad_batches` theo thời gian sử dụng: `GradientAccumulationScheduler`.

```python
from lightning.pytorch.callbacks import GradientAccumulationScheduler

# till 5th epoch, it will accumulate every 8 batches. From 5th epoch
# till 9th epoch it will accumulate every 4 batches and after that no accumulation
# will happen. Note that you need to use zero-indexed epoch keys here
accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
trainer = Trainer(callbacks=accumulator)
```

> NOTE: Not all strategies and accelerators support variable gradient accumulation windows.

### Gradient Clipping

Kỹ thuật "Gradient Clipping" là một cách để xử lý vấn đề của gradient bị vượt quá giới hạn (exploding gradients.) trong quá trình huấn luyện mạng nơ-ron.

Trong quá trình huấn luyện mạng nơ-ron, ta thường sử dụng thuật toán gradient descent để cập nhật các trọng số của mạng dựa trên gradient của hàm mất mát. Gradient là đạo hàm của hàm mất mát đối với các tham số và được sử dụng để điều chỉnh các tham số này sao cho hàm mất mát giảm đi và mô hình học tốt hơn.

Tuy nhiên, đôi khi gradient có thể rất lớn, và điều này có thể gây ra các vấn đề như exploding gradients. Nếu gradient quá lớn, nó có thể làm cho quá trình cập nhật trọng số trở nên không ổn định và mô hình sẽ không học tốt.

Kỹ thuật "Gradient Clipping" giải quyết vấn đề này bằng cách đặt một giới hạn (ngưỡng) trên giá trị tuyệt đối của gradient. Nếu gradient vượt quá giới hạn này sau khi tính toán, nó sẽ được cắt tỉa (clipped) xuống sao cho không vượt quá giới hạn. Điều này giúp đảm bảo rằng gradient luôn có giá trị trong khoảng an toàn và không gây ra sự không ổn định trong quá trình huấn luyện.

Kỹ thuật Gradient Clipping thường được sử dụng khi làm việc với các mô hình sâu (deep learning) hoặc trong các tình huống mà gradient có thể biến đổi rất lớn. Nó giúp duy trì sự ổn định trong quá trình huấn luyện và là một trong những cách để giải quyết vấn đề về gradient vượt quá giới hạn.

```python
# DEFAULT (ie: don't clip)
trainer = Trainer(gradient_clip_val=0)

# clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
trainer = Trainer(gradient_clip_val=0.5)

# clip gradients' maximum magnitude to <=0.5
trainer = Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")
```

[READMORE](https://lightning.ai/docs/pytorch/stable/common/optimization.html#configure-gradient-clipping)

### Stochastic Weight Averaging

Kỹ thuật "`Stochastic Weight Averaging`" (SWA) là một phương pháp trong quá trình huấn luyện mạng nơ-ron để tối ưu hóa mô hình và cải thiện khả năng tổng quát hóa của nó.

Trong quá trình huấn luyện mạng nơ-ron, mục tiêu của chúng ta là tìm ra các trọng số tối ưu để mô hình có khả năng dự đoán tốt trên dữ liệu kiểm tra (tổng quát hóa). Trong thuật ngữ toán học, việc này tương đương với việc tìm trọng số của mạng nơ-ron sao cho hàm mất mát đạt giá trị nhỏ nhất.

Kỹ thuật `Stochastic Weight Averaging` đề xuất sử dụng một phương pháp khác để tối ưu hóa trọng số của mạng. Thay vì sử dụng trọng số cuối cùng sau khi huấn luyện hoàn tất, SWA thực hiện trung bình cộng trọng số từ nhiều vòng lặp khác nhau trong quá trình huấn luyện.

Cụ thể, SWA làm như sau:

1- Huấn luyện mạng nơ-ron bằng cách sử dụng thuật toán `gradient descent` hoặc biến thể của nó để cập nhật trọng số theo mỗi vòng lặp.

2- Trong suốt quá trình huấn luyện, SWA lưu trữ một bản sao của trọng số của mạng (thường được gọi là "trọng số SWA").

3- Sau khi hoàn thành quá trình huấn luyện hoặc sau một số lượng vòng lặp nhất định, trọng số SWA được tính toán bằng cách lấy trung bình cộng của các bộ trọng số đã được lưu trữ.

Trọng số SWA này thường được coi là một mô hình ổn định hơn và có khả năng tổng quát hóa tốt hơn so với trọng số cuối cùng sau khi huấn luyện. Nó có thể giúp giảm thiểu hiện tượng overfitting và cải thiện hiệu suất trên dữ liệu kiểm tra hoặc dữ liệu mới.

SWA thường được sử dụng trong các nhiệm vụ học sâu, đặc biệt là khi có sự cần thiết để tối ưu hóa mô hình để đạt được sự ổn định và khả năng tổng quát hóa.

```python
# Enable Stochastic Weight Averaging using the callback
trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
```

[READMORE](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/)

### Batch Size Finder

Kỹ thuật "`Batch Size Finder`" (Tìm kích thước batch) là một phương pháp trong quá trình huấn luyện mạng nơ-ron để xác định kích thước lý tưởng cho các mini-batch dữ liệu. 

Kích thước batch (batch size) trong quá trình huấn luyện mạng nơ-ron là số lượng mẫu dữ liệu được sử dụng để tính gradient và cập nhật trọng số của mạng trong mỗi vòng lặp (epoch). Lựa chọn kích thước batch có ảnh hưởng đến hiệu suất huấn luyện, tốc độ học tập, và sử dụng bộ nhớ.

Phương pháp `Batch Size Finder` giúp tìm ra giá trị tốt nhất cho kích thước batch mà không cần thử nghiệm nhiều giá trị khác nhau một cách thủ công. Quá trình này thường thực hiện bằng cách sử dụng một quy tắc tự động để xác định kích thước batch tối ưu.

Dưới đây là cách thực hiện `Batch Size Finder`:

1- Bắt đầu với một giá trị ban đầu cho kích thước batch. Thường thì giá trị ban đầu này có thể là một giá trị nhỏ.

2- Huấn luyện mô hình trong một số epoch với kích thước batch này. Sử dụng kích thước batch đã chọn để huấn luyện mô hình trong một số epoch. Sau mỗi epoch, tính toán hiệu suất trên tập dữ liệu kiểm tra hoặc xác thực.

3- So sánh hiệu suất. So sánh hiệu suất của mô hình sau mỗi epoch. Nếu hiệu suất đang tăng hoặc vẫn ổn định, tiếp tục huấn luyện với cùng kích thước batch.

4- Giảm kích thước batch nếu cần. Nếu hiệu suất giảm đi hoặc không thay đổi sau một số epoch, giảm kích thước batch đi một giá trị nhỏ (ví dụ: chia đôi kích thước batch ban đầu) và tiếp tục quá trình huấn luyện.

5- Lặp lại quá trình. Lặp lại bước 2 đến bước 4 cho đến khi hiệu suất không còn cải thiện hoặc đạt đỉnh tốt nhất.

Khi quá trình kết thúc, bạn sẽ có kích thước batch tối ưu cho mô hình của mình, đảm bảo rằng bạn đang sử dụng tối ưu hóa kích thước batch cho hiệu suất tốt nhất trong quá trình huấn luyện. Batch Size Finder giúp tối ưu hóa quá trình này một cách tự động và tiết kiệm thời gian so với việc thử nghiệm nhiều giá trị thủ công.

```python
from lightning.pytorch.tuner import Tuner

# Create a tuner for the trainer
trainer = Trainer(...)
tuner = Tuner(trainer)

# Auto-scale batch size by growing it exponentially (default)
tuner.scale_batch_size(model, mode="power")

# Auto-scale batch size with binary search
tuner.scale_batch_size(model, mode="binsearch")

# Fit as normal with new batch size
trainer.fit(model)

```

```python
# using LightningModule
class LitModel(LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.save_hyperparameters()
        # or
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size | self.hparams.batch_size)


model = LitModel(batch_size=32)
trainer = Trainer(...)
tuner = Tuner(trainer)
tuner.scale_batch_size(model)


# using LightningDataModule
class LitDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.save_hyperparameters()
        # or
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size | self.hparams.batch_size)


model = MyModel()
datamodule = LitDataModule(batch_size=32)

trainer = Trainer(...)
tuner = Tuner(trainer)
tuner.scale_batch_size(model, datamodule=datamodule)
```

### Customizing Batch Size Finder


1- You can also customize the BatchSizeFinder callback to run at different epochs. This feature is useful while fine-tuning models since you can’t always use the same batch size after unfreezing the backbone.

```python
from lightning.pytorch.callbacks import BatchSizeFinder


class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)


trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
trainer.fit(...)
```

2- Run batch size finder for validate/test/predict.

```python
from lightning.pytorch.callbacks import BatchSizeFinder


class EvalBatchSizeFinder(BatchSizeFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_test_start(self, trainer, pl_module):
        self.scale_batch_size(trainer, pl_module)


trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
trainer.test(...)

```

### Learning Rate Finder

Kỹ thuật "`Learning Rate Finder`" (Tìm tốc độ học) là một phương pháp trong quá trình huấn luyện mạng nơ-ron để xác định tốc độ học lý tưởng cho việc cập nhật trọng số của mô hình. 

Tốc độ học (learning rate) trong quá trình huấn luyện mạng nơ-ron là một siêu tham số quan trọng quyết định tốc độ cập nhật của trọng số mô hình dựa trên gradient của hàm mất mát. Lựa chọn tốc độ học phù hợp có thể ảnh hưởng lớn đến hiệu suất huấn luyện và khả năng tổng quát hóa của mô hình.

Phương pháp Learning Rate Finder giúp xác định giá trị tốt nhất cho tốc độ học mà không cần thử nghiệm nhiều giá trị khác nhau một cách thủ công. Quá trình này thường dựa trên việc tăng dần hoặc giảm dần tốc độ học trong quá trình huấn luyện và theo dõi sự biến đổi của hàm mất mát hoặc hiệu suất mô hình.

Dưới đây là cách thực hiện Learning Rate Finder:

1- Chọn một khoảng giá trị tốc độ học. Bắt đầu bằng cách chọn một khoảng giá trị cho tốc độ học. Thông thường, bạn có thể chọn một khoảng như [0.0001, 1.0].

2- Huấn luyện mô hình với các giá trị tốc độ học trong khoảng đã chọn. Bắt đầu với tốc độ học nhỏ nhất và sau mỗi mini-batch hoặc epoch, tăng tốc độ học lên một lượng nhỏ và tiếp tục quá trình huấn luyện.

3- Theo dõi hàm mất mát hoặc hiệu suất. Trong suốt quá trình huấn luyện, theo dõi hàm mất mát trên tập dữ liệu huấn luyện hoặc hiệu suất trên tập dữ liệu kiểm tra hoặc xác thực.

4- Vẽ đồ thị. Vẽ đồ thị của hàm mất mát hoặc hiệu suất dựa trên giá trị tốc độ học. Điều này giúp bạn nhận biết giá trị tốc độ học tối ưu.

5- Chọn tốc độ học lý tưởng. Dựa vào đồ thị, bạn có thể chọn giá trị tốc độ học tốt nhất. Thường thì giá trị tốc độ học lý tưởng nằm ở vùng bắt đầu giảm dần của đồ thị hàm mất mát hoặc tại điểm mất mát nhỏ nhất.

Phương pháp Learning Rate Finder giúp bạn xác định tốc độ học phù hợp và tối ưu nhất cho mô hình của mình một cách tự động và hiệu quả. Điều này giúp cải thiện hiệu suất huấn luyện và tổng quát hóa của mạng nơ-ron.

#### Using Lightning’s built-in LR finder

```python
model = MyModelClass(hparams)
trainer = Trainer()
tuner = Tuner(trainer)

# Run learning rate finder
lr_finder = tuner.lr_find(model)

# Results can be found in
print(lr_finder.results)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# update hparams of the model
model.hparams.lr = new_lr

# Fit model
trainer.fit(model)

```

#### Customizing Learning Rate Finder

```python

from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


trainer = Trainer(callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))])
trainer.fit(...)

```

### Advanced GPU Optimizations

https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html

### Sharing Datasets Across Process Boundaries

Cách để giảm dữ liệu dư thừa khi mà process nào cũng load data lên cho tính toán

```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str):
        self.mnist = MNIST(data_dir, download=True, transform=T.ToTensor())

    def train_loader(self):
        return DataLoader(self.mnist, batch_size=128)


model = Model(...)
datamodule = MNISTDataModule("data/MNIST")

trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_spawn")
trainer.fit(model, datamodule)
```

