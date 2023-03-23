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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.profilers import Profiler, PyTorchProfiler

from visionpod import config


class PodTrainer(L.Trainer):
    """A custom Lightning.LightningTrainer

    # Arguments
        logger: None
        profiler: None
        callbacks: []
        plugins: []
        set_seed: True
        trainer_init_kwargs:
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        profiler: Optional[Profiler] = PyTorchProfiler(dirpath=config.Paths.torch_profiler, filename="profiler"),
        callbacks: Optional[List] = [],
        plugins: Optional[List] = [],
        set_seed: bool = True,
        **trainer_init_kwargs: Dict[str, Any]
    ) -> None:
        if set_seed:
            seed_everything(config.Settings.seed, workers=True)

        if config.System.is_cloud_run:
            profiler = None
            raise UserWarning("Profiler disabled for cloud runs")

        super().__init__(
            logger=logger or TensorBoardLogger(config.Paths.tensorboard, name="logs"),
            profiler=profiler,
            callbacks=callbacks + [ModelCheckpoint(dirpath=config.Paths.checkpoints, filename="model")],
            plugins=plugins,
            **trainer_init_kwargs
        )

    def persist_predictions(self, predictions_dir: Optional[Union[str, Path]] = config.Paths.predictions) -> None:
        """helper method to persist predictions on completion of a training run

        # Arguments
            predictions_dir: the directory path where predictions should be saved to
        """
        self.test(ckpt_path="best", datamodule=self.datamodule)
        predictions = self.predict(self.model, self.datamodule.test_dataloader())
        torch.save(predictions, predictions_dir)
