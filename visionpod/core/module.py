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

import lightning as L
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy
from torchvision.models import vit_b_32 as VisionTransformer
from torchvision.models import ViT_B_32_Weights as Weights
from torchvision.models.vision_transformer import ConvStemConfig


class PodModule(L.LightningModule):
    """A custom PyTorch Lightning LightningModule for torchvision.VisionTransformer.

    # Arguments
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
    """

    def __init__(
        self,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        accuracy_task: str = "multiclass",
        vit_req_image_size: int = 32,
        vit_req_num_classes: int = 10,
        vit_hp_dropout: float = 0.0,
        vit_hp_attention_dropout: float = 0.0,
        vit_hp_norm_layer: Optional[nn.Module] = None,
        vit_hp_conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        vit_init_opt_progress: bool = False,
        vit_init_opt_weights: bool = False,
    ):
        super().__init__()

        if not vit_hp_norm_layer:
            vit_hp_norm_layer = partial(nn.LayerNorm, eps=1e-6)

        vit_kwargs = dict(
            image_size=vit_req_image_size,
            num_classes=vit_req_num_classes,
            dropout=vit_hp_dropout,
            attention_dropout=vit_hp_attention_dropout,
            norm_layer=vit_hp_norm_layer,
            conv_stem_configs=vit_hp_conv_stem_configs,
        )

        self.model = VisionTransformer(
            weights=Weights if vit_init_opt_weights else None,
            progress=vit_init_opt_progress,
            **vit_kwargs,
        )
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.accuracy_task = accuracy_task
        self.num_classes = vit_req_num_classes
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
        """returns a prediction from the trained model"""
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
            acc = accuracy(y_hat, y, task=self.accuracy_task, num_classes=self.num_classes)
            self.log(f"{stage}_acc", acc)
            self.log(f"{stage}_loss", loss)
