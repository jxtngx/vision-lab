### LabModule


```python
visionlab.core.module.LabModule(
    optimizer="Adam",
    lr=0.001,
    accuracy_task="multiclass",
    image_size=32,
    num_classes=10,
    dropout=0.0,
    attention_dropout=0.0,
    norm_layer=None,
    conv_stem_configs=None,
    progress=False,
    weights=False,
    vit_type="b_32",
)
```


A custom PyTorch Lightning LightningModule for torchvision.VisionTransformer.

__Arguments__

- __optimizer__ `str`: "Adam". A valid [torch.optim](https://pytorch.org/docs/stable/optim.html) name.
- __lr__ `float`: 1e-3
- __accuracy_task__ `str`: "multiclass". One of (binary, multiclass, multilabel).
- __image_size__ `int`: 32
- __num_classes__ `int`: 10
- __dropout__ `float`: 0.0
- __attention_dropout__ `float`: 0.0
- __norm_layer__ `torch.nn.modules.module.Module | None`: None
- __conv_stem_configs__ `List[torchvision.models.vision_transformer.ConvStemConfig] | None`: None
- __progress__ `bool`: False
- __weights__ `bool`: False
- __vit_type__ `str`: one of (b_16, b_32, l_16, l_32). Default is b_32.


----

### forward


```python
LabModule.forward(x)
```


calls .forward of a given model flow


----

### training_step


```python
LabModule.training_step(batch)
```


runs a training step sequence in ``.common_step``


----

### test_step


```python
LabModule.test_step(batch, *args)
```


runs a test step sequence in ``.common_step``


----

### validation_step


```python
LabModule.validation_step(batch, *args)
```


runs a validation step sequence in ``.common_step``


----

### predict_step


```python
LabModule.predict_step(batch, batch_idx, dataloader_idx=0)
```


returns predicted logits from the trained model


----

### configure_optimizers


```python
LabModule.configure_optimizers()
```


configures the ``torch.optim`` used in training loop


----

### common_step


```python
LabModule.common_step(batch, stage)
```


consolidates common code for train, test, and validation steps


----
