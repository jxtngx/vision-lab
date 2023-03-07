### ViT_B_16_Parameters


```python
visionpod.core.module.ViT_B_16_Parameters(image_size, num_classes)
```


default parameters for ViT B 16

__Arguments__

- __image_size__ `int`: int. the size of the images.
- __num_classes__ `int`: int. number of classes in the training dataset.

__Notes__

The following args are set by torchvision.vit_b_16
    - patch_size: int = 16
    - num_layers: int = 12
    - num_heads: int = 12
    - hidden_dim: int = 768
    - mlp_dim: int = 3072


----

### ViT_B_16_HyperParameters


```python
visionpod.core.module.ViT_B_16_HyperParameters(
    dropout=0.0, attention_dropout=0.0, representation_size=None, norm_layer=None, conv_stem_configs=None
)
```


default hyperparameters for ViT B 16

__Arguments__

- __dropout__ `float`: float. the likelihood a value will be set to 0.
- __attention_dropout__ `float`: float. the likelihood a value will be set to 0 in the attention heads.
- __representation_size__ `int | None`: int. tbd.
- __norm_layer__ `Callable | None`: callable. a torch normilization layer.
- __conv_stem_configs__ `List[torchvision.models.vision_transformer.ConvStemConfig] | None`: ViT.ConvStemConfig

__Notes__

``norm_layer`` will be set to (partial(nn.LayerNorm, eps=1e-6),) when None


----

### PodModule


```python
visionpod.core.module.PodModule(
    vit_kwargs,
    vit_hyperparameters,
    optimizer="adam",
    lr=0.001,
    accuracy_task="multiclass",
    vit_progress=False,
    vit_weights=None,
)
```


A custom PyTorch Lightning LightningModule.

__Arguments__

- __optimizer__ `str`: a PyTorch Optimizer.
- __lr__ `float`: the learning rate.
- __accuracy_task__ `str`: task for torchmetrics.accuracy.
- __vit_progress__ `bool`: bool. controls the progress bar
- __vit_weights__ `torchvision.models.vision_transformer.ViT_B_16_Weights | None`: Optional[ViT_B_16_Weights].
- __vit_kwargs__ `Dict[str, Any]`: ViT_B_16_Parameters
- __vit_hyperparameters__ `Dict[str, Any]`: ViT_B_16_HyperParameters


----
