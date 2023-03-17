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

from typing import Any, Callable, Dict, Optional

import wandb
from lightning import LightningWork
from lightning.app.utilities.enum import WorkStageStatus
from lightning.pytorch.loggers import WandbLogger

from visionpod import config, PodDataModule, PodModule, PodTrainer


class TrainerWork(LightningWork):
    """trains PodModule with optional HPO Sweep"""

    def __init__(
        self,
        trainer_flags: Dict[str, Any] = config.Trainer.train_flags,
        module_kwargs: Optional[Dict[str, Any]] = config.Args.module_kwargs,
        model_kwargs: Dict[str, Any] = config.Args.model_kwargs,
        model_hypers: Dict[str, Any] = config.Args.model_hyperameters,
        project_name: Optional[str] = config.Settings.projectname,
        sweep: bool = False,
        learned_config: Optional[str] = None,
        fast_train_run: bool = False,
        parallel: bool = False,
        persist_model: bool = False,
        persist_predictions: bool = False,
        predictions_dir=config.Paths.predictions,
        **work_kwargs,
    ) -> None:

        super().__init__(parallel=parallel, cache_calls=True, **work_kwargs)

        if not sweep and not module_kwargs:
            raise ValueError("either module_kwargs must be provided, or sweep must be true")

        if sweep and module_kwargs:
            raise ValueError("set sweep cannot be true if providing module_kwargs")

        if sweep and not learned_config:
            raise ValueError("learned_config must be provided if running a tuned run")

        self.project_name = project_name
        self.sweep = sweep
        self.fast_train_run = fast_train_run
        self.persist_model = persist_model
        self.persist_predictions = persist_predictions
        self.predictions_dir = predictions_dir

        # ._ make private to LightningWork
        self._trainer_flags = trainer_flags
        self._module_kwargs = module_kwargs
        self._model_kwargs = model_kwargs
        self._model_hypers = model_hypers
        # init here to appease App and make private
        self._model = None
        self._datamodule = None
        self._trainer = None
        self._logger = None

        wandb.Api(api_key=config.ExperimentManager.WANDB_API_KEY)

    @property
    def lr(self) -> float | None:
        if hasattr(self, "_sweep_work"):
            return self.best_params["lr"]
        else:
            return self._module_kwargs["lr"]

    @property
    def optimizer(self) -> str | None:
        if hasattr(self, "_sweep_work"):
            return self.best_params["optimizer"]
        else:
            return self._module_kwargs["optimizer"]

    @property
    def dropout(self) -> float | None:
        if hasattr(self, "_sweep_work"):
            return self.best_params["dropout"]
        else:
            return self._model_hypers["dropout"]

    @property
    def attention_dropout(self) -> float | None:
        if hasattr(self, "_sweep_work"):
            return self.best_params["attention_dropout"]
        else:
            return self._model_hypers["attention_dropout"]

    @property
    def norm_layer(self) -> Callable | None:
        if hasattr(self, "_sweep_work"):
            return self.best_params["norm_layer"]
        else:
            return self._model_hypers["norm_layer"]

    @property
    def conv_stem_configs(self):
        if hasattr(self, "_sweep_work"):
            return self.best_params["conv_stem_configs"]
        else:
            return self._model_hypers["conv_stem_configs"]

    @property
    def best_params(self) -> Dict[str, Any] | None:
        if hasattr(self, "_sweep_work"):
            return self._sweep_work.best_params

    @property
    def group_name(self) -> str:
        if hasattr(self, "_sweep_work"):
            return "Tuned Training Runs"
        else:
            if self.fast_train_run:
                return "Fast Training Runs"
            else:
                return "Untuned Training Runs"

    @property
    def run_name(self) -> str:
        if hasattr(self, "_sweep_work"):
            return self._sweep_work.group_name.replace("Sweep", "tuned")
        else:
            if self.fast_train_run:
                return "-".join(["fast-run", wandb.util.generate_id()])
            else:
                return "-".join(["solo-run", wandb.util.generate_id()])

    @property
    def sweep_url(self) -> str | None:
        if hasattr(self, "_trainer"):
            return "/".join([self.entity, self.project_name, "sweeps", self.sweep_id])

    @property
    def entity(self) -> str | None:
        if hasattr(self, "_logger"):
            return self._logger.experiment.entity

    def _persist_model(self) -> None:
        input_sample = self._trainer.datamodule.train_data.dataset[0][0]
        self._trainer.model.to_onnx(config.Paths.model, input_sample=input_sample, export_params=True)

    def _persist_predictions(self, predictions_dir) -> None:
        self._trainer.persist_predictions(predictions_dir=predictions_dir)

    def run(self) -> None:

        self._model = PodModule(
            lr=self.lr,
            optimizer=self.optimizer,
            attention_dropout=self.attention_dropout,
            conv_stem_configs=self.conv_stem_configs,
            dropout=self.dropout,
            norm_layer=self.norm_layer,
            **self._model_kwargs,
        )

        self._datamodule = PodDataModule()

        self._logger = WandbLogger(
            project=self.project_name,
            name=self.run_name,
            group=self.group_name,
            save_dir=config.Paths.wandb_logs,
            id=wandb.util.generate_id(),
            reinit=True,
        )

        self._trainer = PodTrainer(logger=self._logger, **self._trainer_flags)
        self._trainer.fit(model=self._model, datamodule=self._datamodule)

        self._trainer.logger.experiment.finish()

        if self.persist_model:
            self._persist_model()
        if self.persist_predictions:
            self._persist_predictions(self.predictions_dir)

        self.status.stage = WorkStageStatus.SUCCEEDED
