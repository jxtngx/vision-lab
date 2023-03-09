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
from lightning import LightningWork
from lightning.pytorch.loggers import WandbLogger

from visionpod import conf
from visionpod.components.hpo import SweepWork
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.pipeline.datamodule import PodDataModule


class TrainerWork:
    def __init__(
        self,
        experiment_manager: str = "wandb",
        project_name: Optional[str] = None,
    ) -> None:
        self.experiment_manager = experiment_manager
        self.project_name = project_name

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
            return self.module_kwargs["dropout"]

    @property
    def attention_dropout(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["attention_dropout"]
        else:
            return self.module_kwargs["attention_dropout"]

    @property
    def norm_layer(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["norm_layer"]
        else:
            return self.module_kwargs["norm_layer"]

    @property
    def conv_stem_configs(self) -> float:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow.best_params["conv_stem_configs"]
        else:
            return self.module_kwargs["conv_stem_configs"]

    @property
    def group_name(self) -> str:
        if hasattr(self, "_sweep_flow"):
            return self._sweep_flow._sweep_config["name"]
        else:
            return "Solo Training Runs"

    @property
    def run_name(self) -> str | None:
        if hasattr(self, "_sweep_flow"):
            return self.group_name.replace("Sweep", "train")
        else:
            return "-".join(["SoloRun", wandb.util.generate_id()])

    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)

    def persist_predictions(self) -> None:
        self.trainer.persist_predictions()

    def persist_splits(self) -> None:
        self.trainer.datamodule.persist_splits()

    def _fit(self) -> None:

        self.model = PodModule(
            lr=self.lr,
            optimizer=self.optimizer,
            vit_hp_attention_dropout=self.attention_dropout,
            vit_hp_conv_stem_configs=self.conv_stem_configs,
            vit_hp_dropout=self.dropout,
            vit_hp_norm_layer=self.norm_layer,
            **self.module_kwargs,
        )
        self.datamodule = PodDataModule()

        self.logger = WandbLogger(
            project=self.project_name,
            name=self.run_name,
            group=self.group_name,
            save_dir=conf.WANDBPATH,
        )

        self.trainer = PodTrainer(logger=self.logger, **self.trainer_flags)
        self.trainer.fit(model=self.model, datamodule=self.datamodule)

    def run(
        self,
        trainer_flags: Dict[str, Any],
        persist_model: bool = False,
        persist_predictions: bool = False,
        persist_splits: bool = False,
        sweep: bool = False,
        trial_count: Optional[int] = None,
        module_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.trainer_flags = trainer_flags

        if not sweep and not module_kwargs:
            raise ValueError("either module_kwargs must be provided, or sweep must be true")

        if sweep and module_kwargs:
            raise ValueError("set sweep cannot be true if providing module_kwargs")

        if module_kwargs:
            self.module_kwargs = module_kwargs

        if sweep:
            self._sweep_flow = SweepWork(project_name=self.project_name, trial_count=trial_count)
            self._sweep_flow.run(experiment_manager=self.experiment_manager, display_report=False)

        self._fit()

        if persist_model:
            self._train_work.persist_model()
        if persist_predictions:
            self._train_work.persist_predictions()
        if persist_splits:
            self._train_work.persist_splits()
        if issubclass(TrainerWork, LightningWork):
            sys.exit()
