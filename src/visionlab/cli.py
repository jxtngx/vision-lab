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

import typer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from typing_extensions import Annotated

from visionlab import config, CifarDataModule, ViTModule

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]

app = typer.Typer()
docs_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(docs_app, name="docs")
app.add_typer(run_app, name="run")


@app.callback()
def callback() -> None:
    pass


# Docs


@docs_app.command("build")
def build_docs() -> None:
    import shutil

    os.system("mkdocs build")
    shutil.copyfile(src="README.md", dst="docs/index.md")


@docs_app.command("serve")
def serve_docs() -> None:
    os.system("mkdocs serve")


# Run


@run_app.command("dev")
def run_dev():
    datamodule = CifarDataModule()
    model = ViTModule()
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=datamodule)


@run_app.command("trainer")
def run_demo(
    logger: Annotated[str, typer.Option(help="logger to use. one of (`wandb`, `csv`)")] = "csv",
):
    logger = TensorBoardLogger(save_dir=config.Paths.csvlogger, name="tensorboard")

    datamodule = CifarDataModule()
    model = ViTModule()
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        enable_checkpointing=True,
        max_epochs=5,
        callbacks=[
            EarlyStopping(monitor="val-loss", mode="min"),
            ModelCheckpoint(dirpath=config.Paths.trials, filename="model"),
        ],
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule)