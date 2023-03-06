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

from typing import Any, Dict

import lightning as L
import torch.nn.functional as F
from torch import optim
from torchmetrics.functional import accuracy
from torchvision.models import vit_b_16 as VisionTransformer


class PodModule(L.LightningModule):
    """A custom PyTorch Lightning LightningModule.

    # Arguments
        optimizer: a PyTorch Optimizer.
        lr: the learning rate.
        accuracy_task: task for torchmetrics.accuracy.
        num_classes: number of classes.
    """

    def __init__(
        self,
        optimizer: str = "adam",
        lr: float = 1e-3,
        accuracy_task: str = "multiclass",
        num_classes: int = 10,
        vit_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.vit = VisionTransformer(**vit_kwargs)
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.accuracy_task = accuracy_task
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, x):
        return self.vit(x)

    def training_step(self, batch):
        return self._common_step(batch, "training")

    def test_step(self, batch, *args):
        self._common_step(batch, "test")

    def validation_step(self, batch, *args):
        self._common_step(batch, "val")

    def _common_step(self, batch, stage):
        x, y = batch
        print(stage, x.shape)
        y_hat = self.vision_net(x)
        loss = F.cross_entropy(y_hat, y)

        if stage in ["val", "test"]:
            acc = accuracy(y_hat, y, task=self.accuracy_task, num_classes=self.num_classes)
            self.log(f"{stage}_acc", acc)
            self.log(f"{stage}_loss", loss)
        if stage == "training":
            self.log(f"{stage}_loss", loss)
            return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
