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

import sys
from typing import Any, Dict, Optional

import wandb
from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from visionpod import conf
from visionpod.components.hpo import SweepFlow
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.pipeline.datamodule import PodDataModule


class TrainWork:
    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)

    def persist_predictions(self) -> None:
        self.trainer.persist_predictions()

    def persist_splits(self) -> None:
        self.trainer.datamodule.persist_splits()

    def run(
        self,
        project_name: str,
        training_run_name: Optional[str] = None,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        wandb_dir: Optional[str] = conf.WANDBPATH,
    ) -> None:

        self.model = PodModule(lr=lr, optimizer=optimizer)
        self.datamodule = PodDataModule()

        group_name = "Solo Training Runs" if not training_run_name else "Sweep Training Runs"

        if not training_run_name:
            training_run_name = wandb.util.generate_id()

        logger = WandbLogger(
            project=project_name,
            name=training_run_name,
            group=group_name,
            save_dir=wandb_dir,
        )
        trainer_init_kwargs = {
            "max_epochs": 100,
            "callbacks": [
                EarlyStopping(monitor="training_loss", mode="min"),
            ],
        }

        self.trainer = PodTrainer(logger=logger, **trainer_init_kwargs)
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


class TrainFlow:
    def __init__(
        self,
        experiment_manager: str = "wandb",
        project_name: Optional[str] = None,
        trial_count: int = 10,
    ) -> None:
        self.experiment_manager = experiment_manager
        self.project_name = project_name
        self._sweep_flow = SweepFlow(project_name=project_name, trial_count=trial_count)
        self._train_work = TrainWork()

    @property
    def best_params(self) -> Dict[str, Any]:
        return self._sweep_flow.best_params

    @property
    def lr(self) -> float:
        return self._sweep_flow.best_params["lr"]

    @property
    def dropout(self) -> float:
        return self._sweep_flow.best_params["dropout"]

    @property
    def optimizer(self) -> str:
        return self._sweep_flow.best_params["optimizer"]

    @property
    def sweep_group(self) -> str:
        return self._sweep_flow._sweep_config["name"]

    @property
    def run_name(self) -> str:
        return self.sweep_group.replace("Sweep", "train")

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        persist_splits: bool = False,
        sweep: bool = False,
    ) -> None:

        if sweep:
            self._sweep_flow.run(experiment_manager=self.experiment_manager, display_report=False)

        self._train_work.run(
            lr=self.lr,
            optimizer=self.optimizer,
            project_name=self.project_name,
            training_run_name=self.run_name,
        )

        if persist_model:
            self._train_work.persist_model()
        if persist_predictions:
            self._train_work.persist_predictions()
        if persist_splits:
            self._train_work.persist_splits()
        if issubclass(TrainFlow, LightningFlow):
            sys.exit()
