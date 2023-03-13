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
from pathlib import Path

import click

from visionpod import config
from visionpod.cli.utils import common_destructive_flow, make_bug_trainer, teardown
from visionpod.components.train import TrainerWork
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.fabric.bugreport import bugreport
from visionpod.fabric.docs.autogen import PodDocsGenerator
from visionpod.pipeline.datamodule import PodDataModule

FILEPATH = Path(__file__)
package = FILEPATH.parents[1]
project = FILEPATH.parents[2]


@click.group()
def main() -> None:
    pass


@main.command("teardown")
def _teardown() -> None:
    common_destructive_flow([teardown], command_name="teardown")


@main.group("run")
def run() -> None:
    pass


@run.command("bug-report")
def bug_report() -> None:
    bugreport.main()
    print("\n")
    make_bug_trainer()
    trainer = os.path.join(package, "core", "bug_trainer.py")
    run_command = " ".join(["python", trainer, " 2> boring_trainer_error.md"])
    os.system(run_command)
    os.remove(trainer)


@run.command("demo-ui")
def demo_ui() -> None:
    ui = os.path.join(project, "research", "demo", "app.py")
    run_command = f"lightning run app {ui}"
    os.system(run_command)


@main.group("docs")
def docs() -> None:
    pass


@docs.command("build")
def build_docs() -> None:
    PodDocsGenerator.build()


@docs.command("start")
def start_docs() -> None:
    _cwd = os.getcwd()
    try:
        os.chdir(os.path.join(project, "docs-src"))
        os.system("yarn start")
    except KeyboardInterrupt:
        os.chdir(_cwd)


@main.group("trainer")
def trainer() -> None:
    pass


@trainer.group("run")
def trainer_run() -> None:
    pass


@trainer_run.command("fast-dev")
@click.option("--image_size", default=config.Args.model_kwargs["image_size"])
@click.option("--num_classes", default=config.Args.model_kwargs["num_classes"])
def fast_dev_run(image_size, num_classes) -> None:
    model = PodModule(image_size=image_size, num_classes=num_classes)
    datamodule = PodDataModule()
    trainer = PodTrainer(fast_dev_run=True, **config.Trainer.default_flags)
    trainer.fit(model=model, datamodule=datamodule)


@trainer_run.command("fast-train")
@click.option("--em", default="wandb", type=click.Choice(["wandb", "optuna"]))
@click.option("--project-name", default="visionpod")
@click.option("--persist_model", default=False)
@click.option("--persist_predictions", default=True)
@click.option("--persist_splits", default=True)
def fast_train_run(em, project_name, persist_model, persist_predictions, persist_splits) -> None:
    trainer = TrainerWork(
        trainer_flags=config.Trainer.fast_flags,
        experiment_manager=em,
        project_name=project_name,
        sweep=False,
        trial_count=None,
        fast_train_run=True,
    )
    trainer.run(
        persist_model=persist_model,
        persist_predictions=persist_predictions,
        persist_splits=persist_splits,
    )


@trainer_run.command("untuned")
@click.option("--em", default="wandb", type=click.Choice(["wandb", "optuna"]))
@click.option("--project-name", default="visionpod")
@click.option("--persist_model", default=False)
@click.option("--persist_predictions", default=True)
@click.option("--persist_splits", default=True)
def traier_run(em, project_name, persist_model, persist_predictions, persist_splits) -> None:
    flags = config.Trainer.default_flags
    flags["callbacks"] = []
    flags["max_epochs"] = 25
    trainer = TrainerWork(
        trainer_flags=flags,
        experiment_manager=em,
        project_name=project_name,
        sweep=False,
        trial_count=None,
    )
    trainer.run(
        persist_model=persist_model,
        persist_predictions=persist_predictions,
        persist_splits=persist_splits,
    )


@trainer_run.command("sweep")
@click.option("--project-name", default="visionpod")
def sweep_only(project_name) -> None:
    pass


@trainer_run.command("tuned")
@click.option("--em", default="wandb", type=click.Choice(["wandb", "optuna"]))
@click.option("--project-name", default="visionpod")
@click.option("--trial-count", default=10)
@click.option("--persist_model", is_flag=True)
@click.option("--persist_predictions", is_flag=True)
@click.option("--persist_splits", is_flag=True)
@click.option("--image_size", default=config.Args.model_kwargs["image_size"])
@click.option("--num_classes", default=config.Args.model_kwargs["num_classes"])
def tune(
    em, project_name, trial_count, persist_model, persist_predictions, persist_splits, image_size, num_classes
) -> None:
    project_name = "-".join([project_name, em])
    trainer = TrainerWork(experiment_manager=em, project_name=project_name, trial_count=trial_count)
    trainer.run(
        project_name,
        persist_model=persist_model,
        persist_predictions=persist_predictions,
        persist_splits=persist_splits,
    )
