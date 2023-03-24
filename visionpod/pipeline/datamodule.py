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
from typing import Any, Callable, Union

import torch
import torchvision
from lightning.pytorch import LightningDataModule, seed_everything
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from visionpod import config
from visionpod.pipeline.dataset import PodDataset

NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class PodDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Any = PodDataset,
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
        if stage == "fit" or stage is None:
            self.train_data = torch.load(os.path.join(self.data_splits, f"v{self.data_version}-train.pt"))
            self.val_data = torch.load(os.path.join(self.data_splits, f"v{self.data_version}-val.pt"))
        if stage == "test" or stage is None:
            self.test_data = torch.load(os.path.join(self.data_splits, f"v{self.data_version}-test.pt"))

    def _persist_splits(self):
        """saves all splits for reproducibility"""
        seed_everything(config.Settings.seed)
        torchvision.disable_beta_transforms_warning()
        train = self.dataset(self.data_cache, train=True, transform=self.train_transforms)
        val = self.dataset(self.data_cache, train=True, transform=self.test_transforms)
        train_size = int(len(train) * self.train_size)
        val_size = len(train) - train_size
        train_data, _ = random_split(train, lengths=[train_size, val_size])
        _, val_data = random_split(val, lengths=[train_size, val_size])
        test_data = self.dataset(self.data_cache, train=False, transform=self.test_transforms)
        torch.save(train_data, os.path.join(self.data_splits, f"v{self.data_version}-train.pt"))
        torch.save(val_data, os.path.join(self.data_splits, f"v{self.data_version}-val.pt"))
        torch.save(test_data, os.path.join(self.data_splits, f"v{self.data_version}-test.pt"))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)
