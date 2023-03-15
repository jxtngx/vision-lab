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

from lightning import LightningApp, LightningWork
from lightning.pytorch.loggers import WandbLogger

import wandb
from visionpod import config, PodDataModule, PodModule, PodTrainer
from visionpod.components import SweepWork


class TrainerWork(LightningWork):
    def __init__(
        self,
        trainer_flags: Dict[str, Any] = config.Trainer.train_flags,
        module_kwargs: Dict[str, Any] = config.Args.module_kwargs,
        model_kwargs: Dict[str, Any] = config.Args.model_kwargs,
        model_hypers: Dict[str, Any] = config.Args.model_hyperameters,
        sweep: bool = False,
        sweep_kwargs: Optional[Dict[str, Any]] = None,
        trial_count: Optional[int] = None,
        project_name: Optional[str] = "visionpod",
        fast_train_run: bool = False,
        fast_sweep_run: bool = False,
        parallel: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(parallel=parallel, cache_calls=True, **kwargs)

        if not sweep and not module_kwargs:
            raise ValueError("either module_kwargs must be provided, or sweep must be true")

        if sweep and module_kwargs:
            raise ValueError("set sweep cannot be true if providing module_kwargs")

        if sweep:
            self._sweep_work = SweepWork(**sweep_kwargs)

        self.project_name = project_name
        self.sweep = sweep
        self.trial_count = trial_count
        self.fast_train_run = fast_train_run
        self.fast_sweep_run = fast_sweep_run

        # ._ make private to LightningWork
        self._trainer_flags = trainer_flags
        self._module_kwargs = module_kwargs
        self._model_kwargs = model_kwargs
        self._model_hypers = model_hypers

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
        self._trainer = None
        self._logger = None

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
            return self.group_name.replace("Sweep", "tuned")
        else:
            if self.fast_train_run:
                return "-".join(["fast-run", wandb.util.generate_id()])
            else:
                return "-".join(["solo-run", wandb.util.generate_id()])

    def persist_model(self) -> None:
        input_sample = self._trainer.datamodule.train_data.dataset[0][0]
        self._trainer.model.to_onnx(config.Paths.model, input_sample=input_sample, export_params=True)

    def persist_predictions(self, predictions_dir) -> None:
        self._trainer.persist_predictions(predictions_dir=predictions_dir)

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        predictions_dir=config.Paths.predictions,
    ) -> None:

        if self.sweep:
            # should be blocking
            self._sweep_work.run()
            # should only run after above is complete
            self._sweep_work.stop()

        self._logger = WandbLogger(
            project=self.project_name,
            name=self.run_name,
            group=self.group_name,
            save_dir=config.Paths.wandb_logs,
        )

        self._trainer = PodTrainer(logger=self._logger, **self._trainer_flags)
        self._trainer.fit(model=self._model, datamodule=self._datamodule)

        if persist_model:
            self.persist_model()
        if persist_predictions:
            self.persist_predictions(predictions_dir)


app = LightningApp(
    TrainerWork(
        trainer_flags=config.Trainer.fast_flags,
        fast_train_run=True,
        sweep=False,
    )
)
