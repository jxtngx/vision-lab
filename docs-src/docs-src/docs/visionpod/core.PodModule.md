### PodModule


```python
visionpod.core.module.PodModule(optimizer="adam", lr=0.001, accuracy_task="multiclass", num_classes=10)
```


A custom PyTorch Lightning LightningModule.

__Arguments__

- __optimizer__ `str`: a PyTorch Optimizer.
- __lr__ `float`: the learning rate.
- __accuracy_task__ `str`: task for torchmetrics.accuracy.
- __num_classes__ `int`: number of classes.


----
