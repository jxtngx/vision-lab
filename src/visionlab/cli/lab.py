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

import click

from visionlab import config, LabDataModule, LabModule, LabTrainer
from visionlab.cli.utils import common_destructive_flow, make_bug_trainer, teardown
from visionlab.components import SweepWork, TrainerWork
from visionlab.utilities.bugreport import bugreport
from visionlab.utilities.docs.autogen import LabDocsGenerator

PACKAGE = config.Paths.package
PROJECT = config.Paths.project


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
    trainer = os.path.join(PACKAGE, "core", "bug_trainer.py")
    run_command = " ".join(["python", trainer, " 2> boring_trainer_error.md"])
    os.system(run_command)
    os.remove(trainer)


@run.command("demo-ui")
def demo_ui() -> None:
    ui = os.path.join(PROJECT, "research", "demo", "app.py")
    run_command = f"lightning run app {ui}"
    os.system(run_command)


@main.group("docs")
def docs() -> None:
    pass


@docs.command("build")
def build_docs() -> None:
    LabDocsGenerator.build()


@docs.command("start")
def start_docs() -> None:
    _cwd = os.getcwd()
    try:
        os.chdir(os.path.join(PROJECT, "docs-src"))
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
@click.option("--image_size", default=config.Module.model_kwargs["image_size"])
@click.option("--num_classes", default=config.Module.model_kwargs["num_classes"])
def fast_dev(image_size, num_classes) -> None:
    model = LabModule(image_size=image_size, num_classes=num_classes)
    datamodule = LabDataModule()
    trainer = LabTrainer(fast_dev_run=True, **config.Trainer.fast_flags)
    trainer.fit(model=model, datamodule=datamodule)


@trainer_run.command("fast-train")
@click.option("--project-name", default="visionlab")
@click.option("--persist_model", default=False)
@click.option("--persist_predictions", default=True)
def fast_train(project_name, persist_model, persist_predictions) -> None:
    trainer = TrainerWork(
        trainer_flags=config.Trainer.fast_flags,
        project_name=project_name,
        sweep=False,
        trial_count=None,
        fast_train_run=True,
    )
    trainer.run(persist_model=persist_model, persist_predictions=persist_predictions)


@trainer_run.command("untuned")
@click.option("--project-name", default="visionlab")
@click.option("--persist_model", default=False)
@click.option("--persist_predictions", default=True)
def untuned(project_name, persist_model, persist_predictions) -> None:
    flags = config.Trainer.train_flags
    flags["callbacks"] = []
    flags["max_epochs"] = 25
    trainer = TrainerWork(
        trainer_flags=flags,
        project_name=project_name,
        sweep=False,
        trial_count=None,
    )
    trainer.run(persist_model=persist_model, persist_predictions=persist_predictions)


@trainer_run.command("fast-sweep")
@click.option("--project-name", default="visionlab")
@click.option("--persist_model", default=False)
@click.option("--persist_predictions", default=False)
def fast_sweep(project_name, persist_model, persist_predictions) -> None:
    trainer = SweepWork(
        project_name=project_name,
        trainer_init_kwargs=config.Trainer.fast_flags,
    )
    trainer.run()


@trainer_run.command("tuned")
@click.option("--project-name", default="visionlab")
@click.option("--trial-count", default=10)
@click.option("--persist_model", is_flag=True)
@click.option("--persist_predictions", is_flag=True)
@click.option("--image_size", default=config.Module.model_kwargs["image_size"])
@click.option("--num_classes", default=config.Module.model_kwargs["num_classes"])
def tuned(
    project_name,
    trial_count,
    persist_model,
    persist_predictions,
    image_size,
    num_classes,
) -> None:
    trainer = TrainerWork(project_name=project_name, trial_count=trial_count)
    trainer.run(
        project_name,
        persist_model=persist_model,
        persist_predictions=persist_predictions,
    )
