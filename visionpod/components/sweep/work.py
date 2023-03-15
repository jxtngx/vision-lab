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
from typing import Any, Dict, Optional

from lightning import LightningApp, LightningWork
from lightning.pytorch.loggers import WandbLogger

import wandb
from visionpod import config
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.pipeline.datamodule import PodDataModule


class SweepWork(LightningWork):
    """manages hyperparameter tuning with W&B Sweeps"""

    def __init__(
        self,
        wandb_save_dir: Optional[str] = config.Paths.wandb_logs,
        project_name: Optional[str] = "visionpod",
        trial_count: int = 10,
        sweep_config: Dict[str, Any] = config.Sweep.config,
        trainer_init_kwargs: Dict[str, Any] = config.Trainer.sweep_flags,
        parallel: bool = False,
        **kwargs,
    ):

        super().__init__(parallel=parallel, cache_calls=True, **kwargs)

        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.sweep_config = sweep_config
        self.trainer_init_kwargs = trainer_init_kwargs
        self.trial_number = 1
        self.trial_count = trial_count
        self._datamodule = PodDataModule()
        self._wandb_api = wandb.Api()
        self.run_sentinel = 0
        self._trainer = None
        self.sweep_id = None
        self.sweep_name = None

    @property
    def wandb_settings(self) -> Dict[str, Any] | None:
        if hasattr(self, "trainer"):
            return self._trainer.logger.experiment.settings

    @property
    def sweep_url(self) -> str | None:
        if hasattr(self, "trainer"):
            return "/".join([self.entity, self.project_name, "sweeps", self.sweep_id])

    @property
    def entity(self) -> str | None:
        if hasattr(self, "trainer"):
            return self._trainer.logger.experiment.entity

    @property
    def best_params(self) -> Dict[str, Any] | None:
        if hasattr(self, "trainer"):
            return self._wandb_api.sweep(self.sweep_url).best_run().config

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

        self._trainer = PodTrainer(
            logger=logger,
            **self.trainer_init_kwargs,
        )

        self._trainer.logger.log_hyperparams(hyperparameters)

        self._trainer.fit(model=model, datamodule=self._datamodule)

        self.trial_number += 1

        return self._trainer.callback_metrics["val_acc"].item()

    def run(self) -> float:
        print(self.run_sentinel)
        if self.run_sentinel == 0:
            self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.project_name)
            self.sweep_name = "-".join(["Sweep", self.sweep_id])
            self.sweep_config.update({"name": self.sweep_name})
        wandb.agent(self.sweep_id, function=self.objective, count=self.trial_count)
        # should only run after agent is done
        if self.run_sentinel == self.trial_count:
            os.system(f"wandb sweep --stop {self.entity}/{self.project_name}/{self.sweep_id}")
        self.run_sentinel += 1


app = LightningApp(SweepWork(**config.Sweep.work_kwargs))
