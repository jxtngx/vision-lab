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

import json
import os
from typing import Any, Dict, Optional

import wandb
from lightning import CloudCompute, LightningWork
from lightning.app.utilities.enum import WorkStageStatus
from lightning.pytorch.loggers import WandbLogger

from visionpod import config, PodDataModule, PodModule, PodTrainer


class SweepWork(LightningWork):
    """manages hyperparameter tuning with W&B Sweeps"""

    def __init__(
        self,
        wandb_save_dir: Optional[str] = config.Paths.wandb_logs,
        project_name: Optional[str] = config.Settings.projectname,
        sweep_config: Dict[str, Any] = config.Sweep.config,
        trainer_init_flags: Dict[str, Any] = config.Sweep.trainer_flags,
        model_kwargs: Optional[Dict[str, Any]] = None,
        trial_count: int = 10,
        parallel: bool = False,
        machine: str = "default",
        idle_timeout: int = 60,
        interruptible: bool = False,
        **kwargs,
    ):

        try:  # if interruptible not supported error
            cloud_compute = CloudCompute(name=machine, idle_timeout=idle_timeout, interruptible=interruptible)
        except ValueError:
            cloud_compute = CloudCompute(name=machine, idle_timeout=idle_timeout, interruptible=False)

        super().__init__(
            parallel=parallel,
            cache_calls=True,
            cloud_compute=cloud_compute,
            **kwargs,
        )

        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.sweep_config = sweep_config
        self.trainer_init_flags = trainer_init_flags
        self._model_kwargs = model_kwargs
        self.trial_count = trial_count
        self.trial_number = 1
        # must be instantiated in __init__
        # ._ makes a non JSON-serializable attribute private to LightningWork
        self.sweep_id = None
        self.sweep_name = None
        self._datamodule = None
        self._wandb_api = None
        self._trainer = None

    @property
    def wandb_settings(self) -> Dict[str, Any] | None:
        if self._trainer is not None:
            return self._trainer.logger.experiment.settings

    @property
    def sweep_url(self) -> str | None:
        if self._trainer is not None:
            return "/".join([self.entity, self.project_name, "sweeps", self.sweep_id])

    @property
    def entity(self) -> str | None:
        if self._trainer is not None:
            return self._trainer.logger.experiment.entity

    @property
    def group_name(self) -> str:
        return self.sweep_config["name"]

    @property
    def best_params(self) -> Dict[str, Any] | None:
        if self._trainer is not None:
            return self._wandb_api.sweep(self.sweep_url).best_run().config

    def log_results(self) -> None:
        fp = os.path.join(config.Paths.tuned_configs, f"{self.sweep_name.replace('Sweep', 'sweep')}.json")
        with open(fp, "w") as filepath:
            json.dump(self.best_params, filepath, indent=4, sort_keys=True)

    def objective(self) -> float:

        logger = WandbLogger(
            project=self.project_name,
            name="-".join(["sweep", self.sweep_id, "trial", str(self.trial_number)]),
            group=self.group_name,
            save_dir=self.wandb_save_dir,
        )

        learnable_parameters = dict(
            optimizer=wandb.config.optimizer,
            lr=wandb.config.lr,
            dropout=wandb.config.dropout,
            attention_dropout=wandb.config.attention_dropout,
        )

        module_payload = {**self._model_kwargs, **learnable_parameters}

        model = PodModule(**module_payload)

        self._trainer = PodTrainer(
            logger=logger,
            **self.trainer_init_flags,
        )

        self._trainer.logger.log_hyperparams(learnable_parameters)

        self._trainer.fit(model=model, datamodule=self._datamodule)

        self.trial_number += 1

        return self._trainer.callback_metrics["val_acc"].item()

    def run(self) -> None:
        self._datamodule = PodDataModule()
        self._wandb_api = wandb.Api(api_key=config.ExperimentManager.WANDB_API_KEY)
        os.environ[wandb.env.DIR] = self.wandb_save_dir
        self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.project_name)
        self.sweep_name = "-".join(["Sweep", self.sweep_id])
        self.sweep_config.update({"name": self.sweep_name})
        wandb.agent(self.sweep_id, function=self.objective, count=self.trial_count)
        os.system(f"wandb sweep --stop {self.entity}/{self.project_name}/{self.sweep_id}")
        wandb.finish()
        self.log_results()
        self.status.stage = WorkStageStatus.SUCCEEDED
