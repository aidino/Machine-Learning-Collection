# FIND BOTTLENECKS IN YOUR CODE (BASIC)

## Find training loop bottlenecks

```python
trainer = Trainer(profiler="simple")
```

## Profile the time within every function

```python
trainer = Trainer(profiler="advanced")

# or 

from lightning.pytorch.profilers import AdvancedProfiler

profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
trainer = Trainer(profiler=profiler)
```

## Measure accelerator usage

```python
from lightning.pytorch.callbacks import DeviceStatsMonitor

trainer = Trainer(callbacks=[DeviceStatsMonitor()])
```

CPU metrics will be tracked by default on the CPU accelerator. To enable it for other accelerators set `DeviceStatsMonitor(cpu_stats=True)`. To disable logging CPU metrics, you can specify `DeviceStatsMonitor(cpu_stats=False)`.