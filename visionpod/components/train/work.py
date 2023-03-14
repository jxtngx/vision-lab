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

from lightning import LightningWork
from lightning.pytorch.loggers import WandbLogger

import wandb
from visionpod import config
from visionpod.components.hpo import SweepWork
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.pipeline.datamodule import PodDataModule


class TrainerWork:
    def __init__(
        self,
        trainer_flags: Dict[str, Any] = config.Trainer.train_flags,
        module_kwargs: Dict[str, Any] = config.Args.module_kwargs,
        model_kwargs: Dict[str, Any] = config.Args.model_kwargs,
        model_hypers: Dict[str, Any] = config.Args.model_hyperameters,
        sweep: bool = False,
        trial_count: Optional[int] = None,
        experiment_manager: str = "wandb",
        project_name: Optional[str] = None,
        fast_train_run: bool = False,
    ) -> None:

        if not sweep and not module_kwargs:
            raise ValueError("either module_kwargs must be provided, or sweep must be true")

        if sweep and module_kwargs:
            raise ValueError("set sweep cannot be true if providing module_kwargs")

        self.trainer_flags = trainer_flags
        self.module_kwargs = module_kwargs
        self.model_hypers = model_hypers
        self.model_kwargs = model_kwargs
        self.experiment_manager = experiment_manager
        self.project_name = project_name
        self.sweep = sweep
        self.trial_count = trial_count
        self.fast_train_run = fast_train_run

        self.model = PodModule(
            lr=self.lr,
            optimizer=self.optimizer,
            attention_dropout=self.attention_dropout,
            conv_stem_configs=self.conv_stem_configs,
            dropout=self.dropout,
            norm_layer=self.norm_layer,
            **self.model_kwargs,
        )

        self.datamodule = PodDataModule()

        self.trainer = PodTrainer(**self.trainer_flags)

    @property
    def best_params(self) -> Dict[str, Any]:
        return self._sweep_flow.best_params

    @property
    def lr(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["lr"]
        else:
            return self.module_kwargs["lr"]

    @property
    def optimizer(self) -> str:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["optimizer"]
        else:
            return self.module_kwargs["optimizer"]

    @property
    def dropout(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["dropout"]
        else:
            return self.model_hypers["dropout"]

    @property
    def attention_dropout(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["attention_dropout"]
        else:
            return self.model_hypers["attention_dropout"]

    @property
    def norm_layer(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["norm_layer"]
        else:
            return self.model_hypers["norm_layer"]

    @property
    def conv_stem_configs(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["conv_stem_configs"]
        else:
            return self.model_hypers["conv_stem_configs"]

    @property
    def group_name(self) -> str:
        if hasattr(self, "_sweep_flow"):
            return "Tuned Training Runs"
        else:
            if self.fast_train_run:
                return "Fast Training Runs"
            else:
                return "Untuned Training Runs"

    @property
    def run_name(self) -> str:
        if hasattr(self, "_sweep_flow"):
            return self.group_name.replace("Sweep", "tuned")
        else:
            if self.fast_train_run:
                return "-".join(["fast-run", wandb.util.generate_id()])
            else:
                return "-".join(["solo-run", wandb.util.generate_id()])

    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(config.Paths.model, input_sample=input_sample, export_params=True)

    def persist_predictions(self, predictions_dir) -> None:
        self.trainer.persist_predictions(predictions_dir=predictions_dir)

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        predictions_dir=config.Paths.predictions,
    ) -> None:

        if self.sweep:
            self._sweep_flow = SweepWork(project_name=self.project_name, trial_count=self.trial_count)
            self._sweep_flow.run()

        self.logger = WandbLogger(
            project=self.project_name,
            name=self.run_name,
            group=self.group_name,
            save_dir=config.Paths.wandb_logs,
        )
        self.trainer.logger = self.logger
        self.trainer.fit(model=self.model, datamodule=self.datamodule)

        if persist_model:
            self.persist_model()
        if persist_predictions:
            self.persist_predictions(predictions_dir)
        if issubclass(TrainerWork, LightningWork):
            sys.exit()
