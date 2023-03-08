### PodModule


```python
visionpod.core.module.PodModule(
    optimizer="Adam",
    lr=0.001,
    accuracy_task="multiclass",
    vit_req_image_size=32,
    vit_req_num_classes=10,
    vit_hp_dropout=0.0,
    vit_hp_attention_dropout=0.0,
    vit_hp_norm_layer=None,
    vit_opt_conv_stem_configs=None,
    vit_init_opt_progress=False,
    vit_init_opt_weights=False,
)
```


A custom PyTorch Lightning LightningModule for torchvision.VisionTransformer.

__Args__

optimizer: "Adam". A valid [torch.optim](https://pytorch.org/docs/stable/optim.html) name.
lr: 1e-3
accuracy_task: "multiclass". One of (binary, multiclass, multilabel).
vit_req_image_size: 32
vit_req_num_classes: 10
vit_hp_dropout: 0.0
vit_hp_attention_dropout: 0.0
vit_hp_norm_layer: None
vit_opt_conv_stem_configs: None
vit_init_opt_progress: False
vit_init_opt_weights: False


----

### forward


```python
PodModule.forward(x)
```


calls .forward of a given model flow


----

### training_step


```python
PodModule.training_step(batch)
```


runs a training step sequence in ``.common_step``


----

### test_step


```python
PodModule.test_step(batch, *args)
```


runs a test step sequence in ``.common_step``


----

### validation_step


```python
PodModule.validation_step(batch, *args)
```


runs a validation step sequence in ``.common_step``


----

### predict_step


```python
PodModule.predict_step(batch, batch_idx, dataloader_idx=0)
```


returns a prediction from the trained model


----

### configure_optimizers


```python
PodModule.configure_optimizers()
```


configures the ``torch.optim`` used in training loop


----

### common_step


```python
PodModule.common_step(batch, stage)
```


consolidates common code for train, test, and validation steps


----
