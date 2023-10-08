# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List, Optional

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy
from torchvision import models
from torchvision.models import ViT_B_32_Weights as Weights
from torchvision.models.vision_transformer import ConvStemConfig


class LabModule(pl.LightningModule):
    """A custom PyTorch Lightning LightningModule for torchvision.VisionTransformer.

    Args:
        optimizer: "Adam". A valid [torch.optim](https://pytorch.org/docs/stable/optim.html) name.
        lr: 1e-3
        accuracy_task: "multiclass". One of (binary, multiclass, multilabel).
        image_size: 32
        num_classes: 10
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
        num_classes: int = 10,
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
            weights=Weights if weights else None,
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
        """runs a training step sequence in ``.common_step``"""
        return self.common_step(batch, "training")

    def test_step(self, batch, *args):
        """runs a test step sequence in ``.common_step``"""
        self.common_step(batch, "test")

    def validation_step(self, batch, *args):
        """runs a validation step sequence in ``.common_step``"""
        self.common_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """returns predicted logits from the trained model"""
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        """configures the ``torch.optim`` used in training loop"""
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def common_step(self, batch, stage):
        """consolidates common code for train, test, and validation steps"""
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        if stage == "training":
            self.log(f"{stage}_loss", loss)
            return loss
        if stage in ["val", "test"]:
            acc = accuracy(y_hat.argmax(dim=-1), y, task=self.accuracy_task, num_classes=self.num_classes)
            self.log(f"{stage}_acc", acc)
            self.log(f"{stage}_loss", loss)
