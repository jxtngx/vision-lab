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

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import optuna
from lightning import LightningWork
from lightning.pytorch.loggers import WandbLogger
from optuna.trial import FrozenTrial, Trial, TrialState
from rich.console import Console
from rich.table import Table

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
        optuna_study_name: Optional[str] = None,
        trial_count: int = 10,
        experiment_manager: str = "wandb",
        sweep_config: Dict[str, Any] = config.Sweep.config,
        trainer_init_kwargs: Dict[str, Any] = config.Trainer.sweep_flags,
    ):
        self.experiment_manager = experiment_manager
        self.project_name = project_name
        self.optuna_study_name = optuna_study_name
        self.wandb_save_dir = wandb_save_dir
        self.sweep_config = sweep_config
        self.trainer_init_kwargs = trainer_init_kwargs

        if self.experiment_manager == "wandb":
            self.trial_number = 1
            self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=project_name)

        if self.experiment_manager == "optuna":
            self.sweep_id = wandb.util.generate_id()
            self.optuna_study = optuna.create_study(direction="maximize", study_name=self.optuna_study_name)

        self.sweep_name = "-".join(["Sweep", self.sweep_id])
        self.sweep_config.update({"name": self.sweep_name})
        self.datamodule = PodDataModule()
        self.trial_count = trial_count

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
    def optuna_trials(self) -> List[FrozenTrial]:
        return self._optuna_study.trials

    @property
    def optuna_pruned_trial(self) -> List[FrozenTrial]:
        return self._optuna_study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    @property
    def optuna_complete_trials(self) -> List[FrozenTrial]:
        return self._optuna_study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    @property
    def optuna_best_trial(self) -> FrozenTrial:
        return self._optuna_study.best_trial

    @property
    def best_params(self):
        if self.experiment_manager == "wandb":
            return self._wandb_api.sweep(self.sweep_url).best_run().config
        if self.experiment_manager == "optuna":
            return self.optuna_study.best_params

    @property
    def artifact_path(self) -> str:
        """helps to sync wandb and optuna directory names for logs"""
        log_dir = self.wandb_settings.log_user or self.wandb_settings.log_internal

        if log_dir:
            log_dir = os.path.dirname(log_dir.replace(os.getcwd(), "."))

        return str(log_dir).split(os.sep)[-2]

    def _set_artifact_dir(self) -> None:
        """sets optuna log file
        Note:
            borrowed from Optuna
            see https://github.com/optuna/optuna/blob/fd841edc732124961113d1915ee8b7f750a0f04c/optuna/cli.py#L1026
        """

        root_logger = logging.getLogger("optuna")
        root_logger.setLevel(logging.DEBUG)

        artifact_dir = os.path.join(config.Paths.optuna, self.artifact_path)

        if not os.path.isdir(artifact_dir):
            os.mkdir(artifact_dir)

        file_handler = logging.FileHandler(filename=os.path.join(config.Paths.optuna, artifact_dir, "optuna.log"))
        file_handler.setFormatter(optuna.logging.create_default_formatter())
        root_logger.addHandler(file_handler)

    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(config.Paths.model, input_sample=input_sample, export_params=True)

    def persist_predictions(self) -> None:
        self.trainer.persist_predictions()

    def persist_splits(self) -> None:
        self.trainer.datamodule.persist_splits()

    def _optuna_objective(self, trial: Trial) -> float:
        """
        Note:
            see:
            - https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
            - https://github.com/nzw0301/optuna-wandb/blob/main/part-1/wandb_optuna.py
            - https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893
            - PyTorch with Optuna (by PyTorch) https://youtu.be/P6NwZVl8ttc
        """
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

        model = PodModule(optimizer=optimizer, lr=lr)

        config = dict(trial.params)
        config["trial.number"] = trial.number

        if hasattr(self, "trainer"):
            # stops previous wandb run so that a new run can be initialized on new trial
            # also helps to avoid hanging process in wandb sdk
            # i.e. if this is called after self.trainer.fit,
            # a key error is encountered in wandb.sdk on the final trial
            # and wandb does not finish, and does not return control to TrialFlow
            self.trainer.logger.experiment.finish()

        self.trainer = PodTrainer(
            logger=WandbLogger(
                project=self.project_name,
                name="-".join(["sweep", self.sweep_id, "trial", str(trial.number)]),
                group=self.sweep_config["name"],
                save_dir=self.wandb_save_dir,
                config=config,
            ),
            **self.trainer_init_kwargs,
        )

        # set optuna logs dir
        self._set_artifact_dir()
        # logs hyperparameters to logs/wandb_logs/wandb/{run_name}/files/config.yaml
        hyperparameters = dict(optimizer=optimizer, lr=lr)
        self.trainer.logger.log_hyperparams(hyperparameters)

        self.trainer.fit(model=model, datamodule=self.datamodule)

        return self.trainer.callback_metrics["val_acc"].item()

    def _wandb_objective(self) -> float:
        logger = WandbLogger(
            project=self.project_name,
            name="-".join(["sweep", self.sweep_id, "trial", str(self.trial_number)]),
            group=self.sweep_config["name"],
            save_dir=self.wandb_save_dir,
        )

        lr = wandb.config.lr
        optimizer = wandb.config.optimizer

        model = PodModule(optimizer=optimizer, lr=lr)

        self.trainer = PodTrainer(
            logger=logger,
            **self.trainer_init_kwargs,
        )

        # logs hyperparameters to logs/wandb_logs/wandb/{run_name}/files/config.yaml
        hyperparameters = dict(optimizer=optimizer, lr=lr)
        self.trainer.logger.log_hyperparams(hyperparameters)

        self.trainer.fit(model=model, datamodule=self.datamodule)

        self.trial_number += 1

        return self.trainer.callback_metrics["val_acc"].item()

    @staticmethod
    def _display_report(trial_metric_names: List[str], trial_info: List[str]) -> None:
        table = Table(title="Study Statistics")

        for col in trial_metric_names:
            table.add_column(col, header_style="cyan")

        table.add_row(*trial_info)

        console = Console()
        console.print(table, new_line_start=True)

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        persist_splits: bool = False,
        display_report: bool = False,
    ) -> float:

        if self.experiment_manager == "wandb":
            wandb.agent(self.sweep_id, function=self._wandb_objective, count=self.trial_count)

        if self.experiment_manager == "optuna":
            self.optuna_study.optimize(self._optuna_objective, n_trials=self.trial_count, timeout=600)

        if display_report:
            self._display_report(
                trial_metric_names=list(self.best_params.keys()),
                trial_info=list(str(i) for i in self.best_params.values()),
            )
        if persist_model:
            self.persist_model()
        if persist_predictions:
            self.persist_predictions()
        if persist_splits:
            self.persist_splits()
        if issubclass(SweepWork, LightningWork):
            self.stop()

    def stop(self) -> None:
        if self.experiment_manager == "wandb":
            os.system(f"wandb sweep --stop {self.entity}/{self.project_name}/{self.sweep_id}")
        if self.experiment_manager == "optuna":
            self.trainer.logger.experiment.finish()
        sys.exit()
