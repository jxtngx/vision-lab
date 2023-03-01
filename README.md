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

# Lightning Pod Vision

<!-- <img src ="https://img.shields.io/badge/Python-000000.svg?style=for-the-badge&logo=Python&logoColor=white" height="29"/> <img src ="https://img.shields.io/badge/TypeScript-000000.svg?style=for-the-badge&logo=TypeScript&logoColor=white" height="29"/> -->

<img src ="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white" height="30"/>
<br>

<img src ="https://img.shields.io/badge/FastAPI-000000.svg?style=for-the-badge&logo=FastAPI&logoColor=white" height="30"/>
<img src ="https://img.shields.io/badge/W&B-000000.svg?style=for-the-badge&logo=weightsandbiases&logoColor=white" height="30"/>
<img src ="https://img.shields.io/badge/Optuna-000000.svg?style=for-the-badge&logo=target&logoColor=white" height="30"/>

<img src ="https://img.shields.io/badge/Next.js-000000.svg?style=for-the-badge&logo=nextdotjs&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Supabase-000000?style=for-the-badge&logo=supabase&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Prisma-000000?style=for-the-badge&logo=prisma&logoColor=white" height="30"/>

<!-- [![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod-example/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod-example)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod-example.svg?style=shield) -->

</div>

## Overview

The core purpose of this repo is to help [Lightning.ai](https://lightning.ai) users familiarize with the Lightning ecosystem by providing an example image classification product built with Lightning. The _product_ is an intelligent agent behind a visual interface.

It uses [Lightning](https://lightning.ai), [ReactJS](https://reactjs.org) + the [Vercel](https://vercel.com) ecosystem, [FastAPI](https://fastapi.tiangolo.com), [Supabase](https://supabase.com), [Zuplo](https://zuplo.com), and [Prisma](https://www.prisma.io); and is built from [Lightning Pod](https://github.com/JustinGoheen/lightning-pod)'s structural concepts, with the exception that this project replaces _lightning_pod.pages_ with a NextJS app in _pod_ui_.

This is a work in progress, especially the front end.

## Programming Languages and Concepts

**Languages**

- [Python](https://www.python.org): data engineering, and machine learning
- [TypeScript](https://www.typescriptlang.org) and [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript): front-end design and visualization, and select back-end services
- [Rust](https://www.rust-lang.org): exposure via project tooling ([ruff](https://beta.ruff.rs/docs/), [turbo](https://turbo.build)) may require troubleshooting errors

**Data Engineering and Database Management**

- Python + [Supabase-Py](https://supabase.com/docs/reference/python/initializing) + [Postgres](https://supabase.com/docs/guides/database/overview): data acquisition and pushing to Supabase-Postgres

**Model Development and Serving**

- [Lightning](Lightning.ai): developing and serving the model
- [Optuna](https://optuna.readthedocs.io/en/stable/): hyperparameter optimization trials
- [Weights and Biases](https://wandb.ai/site): experiment management
- [FastAPI](https://fastapi.tiangolo.com): developing an API to serve the model with
- [Locust](https://github.com/locustio/locust): API load balance testing

**User Interfaces**

- [Supabase](https://supabase.com): database, storage, and user management
- [Prisma](https://www.prisma.io): Next-generation Node.js and TypeScript ORM
- [TurboRepo](https://turbo.build) + [NextJS](https://nextjs.org) + [React](https://reactjs.org): front end
- [Vercel](https://vercel.com): hosting the front end, and monitoring + analytics

**Domain**

- Topics:
  - Computer Vision
  - Image Classification
- Data Source: torchvision CIFAR10
- Domain Libraries: torchvision
- Model: Linear Encoder Decoder

## Roadmap

The general outline for building this project is:

- data acquistion and storage
- do HPO trials and train from best trial config
- persist model
- build Model serving API with FastAPI
- determine feedback criteria for front end users
- design and build front end
- improve on feedback

## Structure

The structure of the project is:

- `docs-src` is the docusaurus project
- `lightning-app` is the root Lightning Flow and App that will be deployed to Lightning
- `next-app` is the Next + React frontend deployed to Vercel
- `pipeline` handles data engineering for Supabase and Prisma
- `requirements` + `requirements.txt` helps Python related CI/CD jobs
- `research` handles HPO and training
- `tests` are tests for visionpod
- `visionpod` is the python package and deep learning network

visionpod is the primary source code for the default branch. the following modules will be handled on separate production branches:

| Module        | Branch   |
| ------------- | -------- |
| docs-src      | DOCS     |
| lightning-app | LAPP     |
| next-app      | NAPP     |
| pipeline      | PIPELINE |
| research      | RESEARCH |

<!-- ## Setup

You must have Python and NPM installed.

To setup a virtual development environment, in terminal, do:

```sh
python3 -m venv .venv/
source .venv/bin/activate
pip install -e ".[full]"
pre-commit install
deactivate
cd pod_ui
yarn install
cd ..
``` -->
