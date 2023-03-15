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

import os
import sys
from typing import Any, Dict, Optional

from lightning import LightningWork
from lightning.pytorch.loggers import WandbLogger

import wandb
from visionpod import config
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.pipeline.datamodule import PodDataModule


class SweepWork:
    """
    Notes:
        see: https://community.wandb.ai/t/run-best-model-off-sweep/2423
    """

    def __init__(
        self,
        wandb_save_dir: Optional[str] = config.Paths.wandb_logs,
        project_name: Optional[str] = None,
        trial_count: int = 10,
        sweep_config: Dict[str, Any] = config.Sweep.config,
        trainer_init_kwargs: Dict[str, Any] = config.Trainer.sweep_flags,
    ):
        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.sweep_config = sweep_config
        self.trainer_init_kwargs = trainer_init_kwargs
        self.trial_number = 1
        self.trial_count = trial_count

        self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=project_name)
        self.sweep_name = "-".join(["Sweep", self.sweep_id])
        self.sweep_config.update({"name": self.sweep_name})
        self.datamodule = PodDataModule()

        self._wandb_api = wandb.Api()

    @property
    def wandb_settings(self) -> Dict[str, Any]:
        return self.trainer.logger.experiment.settings

    @property
    def sweep_url(self) -> str:
        return "/".join([self.entity, self.project_name, "sweeps", self.sweep_id])

    @property
    def entity(self) -> str:
        return self.trainer.logger.experiment.entity

    @property
    def best_params(self):
        return self._wandb_api.sweep(self.sweep_url).best_run().config

    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(config.Paths.model, input_sample=input_sample, export_params=True)

    def persist_predictions(self) -> None:
        self.trainer.persist_predictions()

    def objective(self) -> float:
        logger = WandbLogger(
            project=self.project_name,
            name="-".join(["sweep", self.sweep_id, "trial", str(self.trial_number)]),
            group=self.sweep_config["name"],
            save_dir=self.wandb_save_dir,
        )

        hyperparameters = dict(
            optimizer=wandb.config.optimizer,
            lr=wandb.config.lr,
            dropout=wandb.config.dropout,
            attention_dropout=wandb.config.attention_dropout,
        )

        model = PodModule(**hyperparameters)

        self.trainer = PodTrainer(
            logger=logger,
            **self.trainer_init_kwargs,
        )

        # logs hyperparameters to logs/wandb_logs/wandb/{run_name}/files/config.yaml

        self.trainer.logger.log_hyperparams(hyperparameters)

        self.trainer.fit(model=model, datamodule=self.datamodule)

        self.trial_number += 1

        return self.trainer.callback_metrics["val_acc"].item()

    def run(self, persist_model: bool = False, persist_predictions: bool = False) -> float:

        wandb.agent(self.sweep_id, function=self.objective, count=self.trial_count)

        if persist_model:
            self.persist_model()
        if persist_predictions:
            self.persist_predictions()
        if issubclass(SweepWork, LightningWork):
            self.stop()

    def stop(self) -> None:
        os.system(f"wandb sweep --stop {self.entity}/{self.project_name}/{self.sweep_id}")
        sys.exit()
