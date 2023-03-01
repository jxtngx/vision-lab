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


import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy


class VisionNet(nn.Module):
    """
    Note:
        see below for example
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-loss-function-and-optimizer
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PodModule(L.LightningModule):
    """a custom PyTorch Lightning LightningModule

    Note:
        PodModule was initially based on examples found in Lightning docs, and was adapted to account
        for the Optuna Trial.
        The flow is as follows:
        [encoder: Sequential(Linear, PReLU, Dropout, Linear), decoder: Sequential(Linear, PReLU, Linear)].

        - for Sequential containers see
        https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential
        - for Linear see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear
        - for PReLU see https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html?highlight=prelu#torch.nn.PReLU
        - for Dropout see https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    """

    def __init__(
        self,
        optimizer: str = "adam",
        lr: float = 1e-3,
        dropout: float = 0.5,
        accuracy_task: str = "multiclass",
        num_classes: int = 10,
    ):
        super().__init__()
        self.vision_net = VisionNet()
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.accuracy_task = accuracy_task
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, x):
        y_hat = self.vision_net(x)
        return y_hat

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
