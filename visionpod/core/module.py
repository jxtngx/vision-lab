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

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import lightning as L
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy
from torchvision.models import vit_b_16 as VisionTransformer
from torchvision.models import ViT_B_16_Weights
from torchvision.models.vision_transformer import ConvStemConfig

DEFUALT_NORM_LAYER = (partial(nn.LayerNorm, eps=1e-6),)


@dataclass
class ViT_B_16_Parameters:
    """default parameters for ViT B 16

    # Arguments
        image_size: int. the size of the images.
        num_classes: int. number of classes in the training dataset.

    # Notes
        The following args are set by torchvision.vit_b_16
            - patch_size: int = 16
            - num_layers: int = 12
            - num_heads: int = 12
            - hidden_dim: int = 768
            - mlp_dim: int = 3072
    """

    image_size: int
    num_classes: int


@dataclass
class ViT_B_16_HyperParameters:
    """default hyperparameters for ViT B 16

    # Arguments
        dropout: float. the likelihood a value will be set to 0.
        attention_dropout: float. the likelihood a value will be set to 0 in the attention heads.
        representation_size: int. tbd.
        norm_layer: callable. a torch normilization layer.
        conv_stem_configs: ViT.ConvStemConfig

    # Notes
        ``norm_layer`` will be set to (partial(nn.LayerNorm, eps=1e-6),) when None
    """

    dropout: float = 0.0
    attention_dropout: float = 0.0
    representation_size: Optional[int] = None
    norm_layer: Optional[Callable] = None
    conv_stem_configs: Optional[List[ConvStemConfig]] = None


class PodModule(L.LightningModule):
    """A custom PyTorch Lightning LightningModule for torchvision.VisionTransformer.

    # Arguments
        optimizer: a PyTorch Optimizer.
        lr: the learning rate.
        accuracy_task: task for torchmetrics.accuracy.
        vit_progress: bool. controls the progress bar
        vit_weights: Optional[ViT_B_16_Weights].
        vit_kwargs: ViT_B_16_Parameters
        vit_hyperparameters: ViT_B_16_HyperParameters
    """

    def __init__(
        self,
        model: str,
        vit_kwargs: Dict[str, Any],
        vit_hyperparameters: Dict[str, Any],
        optimizer: str = "adam",
        lr: float = 1e-3,
        accuracy_task: str = "multiclass",
        vit_progress: bool = False,
        vit_weights: Optional["ViT_B_16_Weights"] = None,
    ):
        super().__init__()
        vit_kwargs.update(**vit_hyperparameters)
        if not vit_kwargs["norm_layer"]:
            vit_kwargs.pop("norm_layer")

        self.vit = VisionTransformer(weights=vit_weights, progress=vit_progress, **vit_kwargs)
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.accuracy_task = accuracy_task
        self.num_classes = vit_kwargs["num_classes"]
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
