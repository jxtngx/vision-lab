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

import torch
import typer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from visionlab import config, CifarDataModule, VisionTransformer

this_file = Path(__file__)
this_studio_idx = [i for i, j in enumerate(this_file.parents) if j.name.endswith("this_studio")][0]
this_studio = this_file.parents[this_studio_idx]
csvlogs = os.path.join(this_studio, "vision-lab", "logs", "csv")

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
    model = VisionTransformer()
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=datamodule)


@run_app.command("trainer", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_trainer(
    devices: str = "auto",
    accelerator: str = "auto",
    strategy: str = "auto",
    max_epochs: int = 10,
    predict: bool = True,
):
    datamodule = CifarDataModule()
    model = VisionTransformer()
    trainer = Trainer(
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=max_epochs,
        enable_checkpointing=True,
        callbacks=[
            EarlyStopping(monitor="val-loss", mode="min"),
            ModelCheckpoint(dirpath=config.Paths.ckpts, filename="model"),
        ],
        logger=CSVLogger(save_dir=config.Paths.logs, name="csv"),
        log_every_n_steps=1,
    )
    trainer.fit(model=model, datamodule=datamodule)

    if predict:
        trainer.test(ckpt_path="best", datamodule=datamodule)
        predictions = trainer.predict(model, datamodule.test_dataloader())
        torch.save(predictions, config.Paths.predictions)
