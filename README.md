# Vision Lab

<!-- # Copyright Justin R. Goheen.
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
# limitations under the License. -->

<!-- zed zed zed -->

## Overview

Vision Lab is a public template for computer vision deep learning research projects using [TorchVision](https://pytorch.org/vision/stable/index.html) and Lightning AI's [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/).

Use Vision Lab to train or finetune the default torchvision Vision Transformer or make it your own by implementing a new model and dataset after cloning the repo.

You can fork Vision Lab with the [use this template](https://github.com/new?template_name=vision-lab&template_owner=JustinGoheen) button.

## Source Module

`visionlab.core` contains code for the Lightning Module and Trainer.

`visionlab.components` contains experiment utilities grouped by purpose for cohesion.

`visionlab.pipeline` contains code for data acquistion and preprocessing, and building a TorchDataset and LightningDataModule.

`visionlab.serve` contains code for model serving APIs built with [FastAPI](https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi).

`visionlab.cli` contains code for the command line interface built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/en/stable/).

`visionlab.pages` contains code for data apps built with [Streamlit](https://streamlit.io/).

`visionlab.config` assists with project, trainer, and sweep configurations.

## Base Requirements and Extras

Vision Lab installs minimal requirements out of the box, and provides extras to make creating robust virtual environments easier. To view the requirements, in [setup.cfg](setup.cfg), see `install_requires` for the base requirements and `options.extras_require` for the available extras.

The recommended install is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Using Vision Lab

Vision Lab also enables use of a CLI named `lab` that is built with [Typer](https://typer.tiangolo.com). This CLI is available in the terminal after install. `lab`'s features can be viewed with:

```sh
lab --help
```

A [fast dev run](https://lightning.ai/docs/pytorch/latest/common/trainer.html#fast-dev-run) cab be ran with:

```sh
lab run dev
```

A longer demo run can be inititated with:

```sh
lab run demo
```

### Weights and Biases

If you have a [Weights and Biases](https://wandb.ai/site) account, you can override the default CSV logger and use wandb with:

```sh
lab run demo --logger wandb
```

### Streamlit

Stay tuned for the Streamlit app!
