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
    vit_init_opt_weights=None,
)
```


A custom PyTorch Lightning LightningModule for torchvision.VisionTransformer.

__Arguments__

- __optimizer__ `str`: str = "Adam",
- __lr__ `float`: float = 1e-3,
- __accuracy_task__ `str`: str = "multiclass",
- __vit_req_image_size__ `int`: int = 32,
- __vit_req_num_classes__ `int`: int = 10,
- __vit_hp_dropout__ `float`: float = 0.0,
- __vit_hp_attention_dropout__ `float`: float = 0.0,
- __vit_hp_norm_layer__ `torch.nn.modules.module.Module | None`: Optional[nn.Module] = None,
- __vit_opt_conv_stem_configs__ `List[torchvision.models.vision_transformer.ConvStemConfig] | None`: Optional[List[ConvStemConfig]] = None,
- __vit_init_opt_progress__ `bool`: bool = False,
- __vit_init_opt_weights__ `torchvision.models.vision_transformer.ViT_B_16_Weights | None`: Optional["ViT_B_16_Weights"] = None,


----
