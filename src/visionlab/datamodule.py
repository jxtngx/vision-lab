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

import multiprocessing
import os
from typing import Callable, Union

import torch
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from visionlab import config

NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class CifarDataModule(pl.LightningDataModule):
    """A custom LightningDataModule"""

    def __init__(
        self,
        dataset: VisionDataset = CIFAR100,
        data_cache: str = config.Paths.dataset,
        data_splits: str = config.Paths.splits,
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
        train_transforms: Callable = config.DataModule.train_transform,
        test_transforms: Callable = config.DataModule.test_transform,
        batch_size: int = config.DataModule.batch_size,
        data_version: int = 0,
        reversion: bool = False,
    ):
        super().__init__()
        self.data_cache = data_cache
        self.data_splits = data_splits
        self.dataset = dataset
        self.train_size = train_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.data_version = data_version
        self.reversion = reversion
        self.data_cache_exists = os.path.isdir(self.data_cache)

    def prepare_data(self) -> None:
        """prepares data for the dataloaders"""
        vfiles = (
            f"v{self.data_version}-train.pt",
            f"v{self.data_version}-val.pt",
            f"v{self.data_version}-test.pt",
        )

        version_exists = any(v in os.listdir(self.data_splits) for v in vfiles)

        if not self.data_cache_exists:
            self.dataset(self.data_cache, download=True)
            self._persist_splits()

        if not version_exists:
            self._persist_splits()

        if version_exists and not self.reversion:
            return

        if self.reversion:
            if version_exists:
                raise ValueError("a split version of the same version number already exists")
            self._persist_splits()

    def setup(self, stage: Union[str, None] = None) -> None:
        """used by trainer to setup the dataset for training and evaluation"""
        if stage == "fit" or stage is None:
            self.train_data = torch.load(os.path.join(self.data_splits, f"v{self.data_version}-train.pt"))
            self.val_data = torch.load(os.path.join(self.data_splits, f"v{self.data_version}-val.pt"))
        if stage == "test" or stage is None:
            self.test_data = torch.load(os.path.join(self.data_splits, f"v{self.data_version}-test.pt"))

    def _persist_splits(self):
        """saves all splits for reproducibility"""
        pl.seed_everything(config.Settings.seed)
        torchvision.disable_beta_transforms_warning()
        dataset = self.dataset(self.data_cache, train=True, transform=self.train_transforms)
        train_data, val_data = random_split(dataset, lengths=[self.train_size, 1 - self.train_size])
        test_data = self.dataset(self.data_cache, train=False, transform=self.test_transforms)
        torch.save(train_data, os.path.join(self.data_splits, f"v{self.data_version}-train.pt"))
        torch.save(val_data, os.path.join(self.data_splits, f"v{self.data_version}-val.pt"))
        torch.save(test_data, os.path.join(self.data_splits, f"v{self.data_version}-test.pt"))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """the dataloader used during training"""
        return DataLoader(self.train_data, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """the dataloader used during testing"""
        return DataLoader(self.test_data, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """the dataloader used during validation"""
        return DataLoader(self.val_data, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)
