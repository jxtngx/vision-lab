from functools import partial
from typing import List, Optional

import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy
from torchvision import models
from torchvision.models.vision_transformer import ConvStemConfig

import lightning.pytorch as pl


class VisionTransformer(pl.LightningModule):
    """A custom PyTorch Lightning LightningModule for torchvision VisionTransformers

    Args:
        optimizer: "Adam". A valid [torch.optim](https://pytorch.org/docs/stable/optim.html) name.
        lr: 1e-3
        accuracy_task: "multiclass". One of (binary, multiclass, multilabel).
        image_size: 32
        num_classes: 100
        dropout: 0.0
        attention_dropout: 0.0
        norm_layer: None
        conv_stem_configs: None
        progress: False
        weights: False
        vit_type: one of (b_16, b_32, l_16, l_32). Default is b_32.
    """

    def __init__(
        self,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        accuracy_task: str = "multiclass",
        image_size: int = 32,
        num_classes: int = 100,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Optional[nn.Module] = None,
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        progress: bool = False,
        weights: bool = False,
        vit_type: str = "b_32",
    ):
        super().__init__()

        if vit_type not in ("b_16", "b_32", "l_16", "l_32"):
            raise ValueError("vit_type must be one of (b_16, b_32, l_16, l_32)")

        if not norm_layer:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if weights:
            weights_name = f"ViT_{vit_type.upper()}_Weights"
            weights = getattr(models, weights_name)

        vit_kwargs = dict(
            image_size=image_size,
            num_classes=num_classes,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )

        vision_transformer = getattr(models, f"vit_{vit_type}")

        self.model = vision_transformer(
            weights=weights,
            progress=progress,
            **vit_kwargs,
        )
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.accuracy_task = accuracy_task
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, x):
        """calls .forward of a given model flow"""
        return self.model(x)

    def training_step(self, batch):
        """runs a training step sequence"""
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("training-loss", loss)
        return loss

    def validation_step(self, batch, *args):
        """runs a validation step sequence"""
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val-loss", loss)
        acc = accuracy(
            y_hat.argmax(dim=-1),
            y,
            task=self.accuracy_task,
            num_classes=self.num_classes,
        )
        self.log("val-acc", acc)

    def test_step(self, batch, *args):
        """runs a test step sequence"""
        x, y = batch
        y_hat = self.model(x)
        acc = accuracy(
            y_hat.argmax(dim=-1),
            y,
            task=self.accuracy_task,
            num_classes=self.num_classes,
        )
        self.log("test-acc", acc)

    def predict_step(self, batch):
        """returns predicted logits from the trained model"""
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        """configures the ``torch.optim`` used in training loop"""
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
