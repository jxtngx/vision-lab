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
from pathlib import Path
from typing import Any, Callable, Union

import torch
from lightning.pytorch import LightningDataModule, seed_everything
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from visionpod import conf
from visionpod.pipeline.dataset import PodDataset

filepath = Path(__file__)
PROJECTPATH = os.getcwd()
NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class PodDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Any = PodDataset,
        data_dir: str = conf.DATASETPATH,
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
        train_transforms: Callable = conf.TRAINTRANSFORMS,
        test_transforms: Callable = conf.TESTTRANSFORM,
        batch_size: int = conf.BATCHSIZE,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.train_size = train_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.dataset(self.data_dir, download=True, train=True)
        self.dataset(self.data_dir, download=True, train=False)

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            seed_everything(conf.GLOBALSEED)
            train = self.dataset(self.data_dir, train=True, transform=self.train_transforms)
            val = self.dataset(self.data_dir, train=True, transform=self.test_transforms)
            train_size = int(len(train) * self.train_size)
            val_size = len(train) - train_size
            self.train_data, _ = random_split(train, lengths=[train_size, val_size])
            _, self.val_data = random_split(val, lengths=[train_size, val_size])
        if stage == "test" or stage is None:
            self.test_data = self.dataset(self.data_dir, train=False, transform=self.test_transforms)

    def persist_splits(self):
        """saves all splits for reproducibility"""
        torch.save(self.train_data, conf.TRAINSPLITPATH)
        torch.save(self.val_data, conf.VALSPLITPATH)
        if not hasattr(self, "test_data"):
            self.setup(stage="test")
        torch.save(self.test_data, conf.TESTSPLITPATH)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size)
