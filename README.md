# Lightning-Pod Vision

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

<div align="center">

<!-- <img src ="https://img.shields.io/badge/Python-000000.svg?style=for-the-badge&logo=Python&logoColor=white" height="29"/> <img src ="https://img.shields.io/badge/TypeScript-000000.svg?style=for-the-badge&logo=TypeScript&logoColor=white" height="29"/> -->

<!-- <img src ="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white" height="30"/>
<br/>
<img src ="https://img.shields.io/badge/FastAPI-000000.svg?style=for-the-badge&logo=FastAPI&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/W&B-000000.svg?style=for-the-badge&logo=weightsandbiases&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Optuna-000000.svg?style=for-the-badge&logo=target&logoColor=white" height="30"/>
<br/>
<img src ="https://img.shields.io/badge/Next.js-000000.svg?style=for-the-badge&logo=nextdotjs&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Supabase-000000?style=for-the-badge&logo=supabase&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Prisma-000000?style=for-the-badge&logo=prisma&logoColor=white" height="30"/> -->

<!-- [![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod-example/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod-example)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod-example.svg?style=shield) -->

</div>

## Overview

The core purpose of this repo is to help [Lightning.ai](https://lightning.ai) users familiarize with the Lightning ecosystem by providing an example image classification product built with Lightning. The _product_ is a pre-trained VisionTransformer served from Lightning Platform.

Another purpose of this project is to create a visual interface with [ReactJS](https://reactjs.org) + the [Vercel](https://vercel.com) ecosystem.

Core services and dependencies are: [Weights and Biases](http://wandb.ai/site), [Supabase](https://supabase.com), [Prisma](https://www.prisma.io), [FastAPI](https://fastapi.tiangolo.com), and [Zuplo](https://zuplo.com).

This is a work in progress, especially the front end.

The docs can be viewed at [visionpod-docs.vercel.app](https://visionpod-docs.vercel.app/)

The NextJS app's progress can be viewed at [visionpod.vercel.app](https://visionpod.vercel.app/)

## Tools and Topics

**Domain**

- Topics:
  - Computer Vision
  - Image Classification
- Data Source: [torchvision CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10)
- Domain Libraries: torchvision
- Model: torchvision [VisionTransformer](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)
- Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

**Languages**

- [Python](https://www.python.org): data engineering, and machine learning
- [TypeScript](https://www.typescriptlang.org) and [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript): front-end design and visualization, and select back-end services
- [Rust](https://www.rust-lang.org): exposure via project tooling ([ruff](https://beta.ruff.rs/docs/), [turbo](https://turbo.build), [prisma-client-py](https://github.com/RobertCraigie/prisma-client-py)) may require troubleshooting errors

**Data Engineering and Database Management**

- [Supabase](https://supabase.com): Postgres database, and bucket storage
- [Prisma](https://www.prisma.io): Next-generation Node.js and TypeScript ORM (with community Python client)

**Model Development and Serving**

- [Lightning](Lightning.ai): developing and serving the model
- [Weights and Biases](https://wandb.ai/site): experiment management
- [FastAPI](https://fastapi.tiangolo.com): developing an API to serve the model with
- [Locust](https://github.com/locustio/locust): API load balance testing

**User Interfaces**

- [Supabase](https://supabase.com): user management
- [TurboRepo](https://turbo.build) + [NextJS](https://nextjs.org) + [React](https://reactjs.org): front end
- [Vercel](https://vercel.com): hosting the front end, and monitoring + analytics

## Structure

The structure of the project is:

- `data` contains data cache, splits, and training run predictions
- `docs-src` is the docusaurus project
- `logs` contains logs generated by experiment managers and profilers
- `models` contains training checkpoints and pre-trained models
- `requirements` + `requirements.txt` helps CI/CD jobs install Python requirements
- `research` contains companion notebooks and a Plotly Dash UI
- `tests` are tests for visionpod
- `ui` is the NextJS frontend deployed to Vercel
- `visionpod` is the Python package, Lightning App, and VisionTransformer

## Setup

You must have [Python](https://www.python.org/downloads/) and [NVM](https://github.com/nvm-sh/nvm#installing-and-updating) installed. See [Installing NVM, Node, and TypeScript](https://visionpod-docs.vercel.app/blog/Installing-NVM-Node-and-TypeScript) for help.

To setup a virtual development environment, in terminal, do:

```sh
python3 -m venv .venv/
source .venv/bin/activate
pip install -e ".[full]"
pre-commit install
deactivate
cd ui
yarn config set nodeLinker node-module
yarn install
cd ..
```

## Usage

Using this template will require accounts for Lightning + Weights and Biases.

After creating a W&B account and installing the development environment, a training run can be ran locally with any of the following command line examples.

If you are on an M1 powered mac, be sure to set the following environment variable to train with MPS:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

then, to train or run a Sweep, do any of the following:

```sh
pod trainer run fast-train
```

a Lightning App version can be ran with:

```sh
cd apps/
lightning run app trainer_app.py
```

or, if you'd prefer to perform a wandb Sweep and then run a tuned trainer, do:

```sh
cd apps/
lightning run app tuned_trainer.py
```

Running any of the above will download the CIFAR10 dataset from torchvision, and cache it to `data/cache`.

Once the run is complete, a prototype UI can be ran locally with:

```sh
cd research/demo
lightning run app app.py
```

## Roadmap

The general outline for building this project is:

- data acquistion and storage
- do HPO trials and train from best trial config
- persist model
- build Model serving API with FastAPI
- determine feedback criteria for front end users
- design and build front end
- improve on feedback
