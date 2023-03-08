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

from visionpod import conf
from visionpod.cli.utils import common_destructive_flow, make_bug_trainer, teardown
from visionpod.components import train
from visionpod.core.module import PodModule
from visionpod.core.trainer import PodTrainer
from visionpod.fabric.bugreport import bugreport
from visionpod.fabric.docs.autogen import PodDocsGenerator
from visionpod.pipeline.datamodule import PodDataModule

FILEPATH = Path(__file__)
PKGPATH = FILEPATH.parents[1]
PROJECTPATH = FILEPATH.parents[2]


@click.group()
def main() -> None:
    pass


@main.command("teardown")
def _teardown() -> None:
    common_destructive_flow([teardown], command_name="teardown")


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
        os.chdir(os.path.join(PROJECTPATH, "docs-src"))
        os.system("yarn start")
    except KeyboardInterrupt:
        os.chdir(_cwd)


@main.command("bug-report")
def bug_report() -> None:
    bugreport.main()
    print("\n")
    make_bug_trainer()
    trainer = os.path.join(PKGPATH, "core", "bug_trainer.py")
    run_command = " ".join(["python", trainer, " 2> boring_trainer_error.md"])
    os.system(run_command)
    os.remove(trainer)


@main.group("trainer")
def trainer() -> None:
    pass


@trainer.command("help")
def help() -> None:
    trainer = os.path.join(PKGPATH, "core", "trainer.py")
    os.system(f"python {trainer} --help")


@trainer.command("fast-dev-run")
@click.option("--image_size", default=conf.IMAGESIZE)
@click.option("--num_classes", default=conf.NUMCLASSES)
def fast_dev_run(image_size, num_classes) -> None:
    model = PodModule(vit_req_image_size=image_size, vit_req_num_classes=num_classes)
    datamodule = PodDataModule()
    trainer = PodTrainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=datamodule)


@trainer.command("sweep-and-train")
@click.option("--em", default="wandb", type=click.Choice(["wandb", "optuna"]))
@click.option("--project-name", default="visionpod")
@click.option("--trial-count", default=10)
@click.option("--persist_model", is_flag=True)
@click.option("--persist_predictions", is_flag=True)
@click.option("--persist_splits", is_flag=True)
@click.option("--image_size", default=conf.IMAGESIZE)
@click.option("--num_classes", default=conf.NUMCLASSES)
def sweep_and_train(
    em, project_name, trial_count, persist_model, persist_predictions, persist_splits, image_size, num_classes
) -> None:
    project_name = "-".join([project_name, em])
    trainer = train.TrainFlow(experiment_manager=em, project_name=project_name, trial_count=trial_count)
    trainer.run(
        project_name,
        persist_model=persist_model,
        persist_predictions=persist_predictions,
        persist_splits=persist_splits,
    )


@trainer.command("train-only")
@click.option("--project-name", default="visionpod")
@click.option("--persist_model", is_flag=True)
@click.option("--persist_predictions", is_flag=True)
@click.option("--persist_splits", is_flag=True)
@click.option("--image_size", default=conf.IMAGESIZE)
@click.option("--num_classes", default=conf.NUMCLASSES)
def train_only(project_name, persist_model, persist_predictions, persist_splits, image_size, num_classes) -> None:
    trainer = train.TrainWork()
    trainer.run(project_name)


@trainer.command("sweep-only")
@click.option("--project-name", default="visionpod")
def sweep_only(project_name) -> None:
    pass
