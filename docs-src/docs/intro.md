---
sidebar_position: 1
---

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

The core purpose of this repo is to help [Lightning.ai](https://lightning.ai) users familiarize with the Lightning ecosystem by providing an example image classification product built with Lightning. The _product_ is a model served via API.

Another purpose of this project is to create a visual interface with [ReactJS](https://reactjs.org) + the [Vercel](https://vercel.com) ecosystem.

Core services and dependencies are: [Supabase](https://supabase.com), [Prisma](https://www.prisma.io), [FastAPI](https://fastapi.tiangolo.com), and [Zuplo](https://zuplo.com).

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
- [Optuna](https://optuna.readthedocs.io/en/stable/): hyperparameter optimization trials
- [Weights and Biases](https://wandb.ai/site): experiment management
- [FastAPI](https://fastapi.tiangolo.com): developing an API to serve the model with
- [Locust](https://github.com/locustio/locust): API load balance testing

**User Interfaces**

- [Supabase](https://supabase.com): user management
- [TurboRepo](https://turbo.build) + [NextJS](https://nextjs.org) + [React](https://reactjs.org): front end
- [Vercel](https://vercel.com): hosting the front end, and monitoring + analytics

## Structure

The structure of the project is:

- `docs-src` is the docusaurus project
- `next-app` is the Next + React frontend deployed to Vercel
- `requirements` + `requirements.txt` helps CI/CD jobs install Python requirements
- `research` location of data cache, experiment logs, companion notebooks, checkpoints, and pre-trained model
- `tests` are tests for visionpod
- `visionpod` is the python package, Lightning App, and VisionTransformer.

## Setup

You must have [Python](https://www.python.org/downloads/) and [NVM](https://github.com/nvm-sh/nvm#installing-and-updating) installed. See [Installing NVM, Node, and TypeScript](https://visionpod-docs.vercel.app/blog/Installing-NVM-Node-and-TypeScript) for help.

To setup a virtual development environment, in terminal, do:

```sh
python3 -m venv .venv/
source .venv/bin/activate
pip install -e ".[full]"
pre-commit install
deactivate
cd next-app
yarn config set nodeLinker node-module
yarn install
cd ..
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

## Learning Resources

### JavaScript, TypeScript, and React

All links shown below are for free classes offered at Codecademy. Please note these classes can be accessed and completed for free; however, the completion certificate is offered only for paid accounts.

- [Learn JavaScript Course](https://www.codecademy.com/learn/introduction-to-javascript)
- [Learn TypeScript Course](https://www.codecademy.com/learn/learn-typescript)
- [Learn React Course](https://www.codecademy.com/learn/react-101)

### Machine Learning and Basic SWE

#### Reviewing Source Code

The following three videos were created by Lightning's Thomas Chaton; the videos are extremely helpful in learning how to use code search features in VS Code to navigate a project's source code, enabling a deeper understanding of what is going on under the hood of someone else's code.

> these videos were created before PyTorch Lightning was moved into the Lightning Framework mono repo

- [Lightning Codebase Deep Dive 1](https://youtu.be/aEeh9ucKUkU)
- [Lightning Codebase Deep Dive 2](https://youtu.be/NEpRYqdsm54)
- [Lightning Codebase Deep Dive 3](https://youtu.be/x4d4RDNJaZk)

#### General Engineering and Tools

Lightning's founder, and their lead educator have created a series of short videos called [Lightning Bits](https://lightning.ai/pages/ai-education/#bits) for beginners who need guides for using IDEs, git, and terminal.

A long standing Python community resource has been [The Hitchhiker's Guide to Python](https://docs.python-guide.org). The "guide exists to provide both novice and expert Python developers a best practice handbook for the installation, configuration, and usage of Python on a daily basis".

[VS Code](https://code.visualstudio.com/docs) and [PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) IDEs have each provided great docs for their users. My preference is VS Code - though PyCharm does have its benefits and is absolutely a suitable alternative to VS Code. I especially like VS Code's [integrations for PyTorch and tensorboard](https://code.visualstudio.com/docs/datascience/pytorch-support). I pair [Gitkraken](https://www.gitkraken.com) and [GitLens](https://www.gitkraken.com/gitlens) with VS Code to manage my version control and contributions.

#### Data Analysis

Wes McKinney, creator of Pandas and founder of Voltron Data (responsible for Ibis, Apache Arrow etc) has released his third edition of [Python for Data Analysis](https://wesmckinney.com/book/) in an open access format.

#### Intro to Artificial Intelligence and Mathematics for Machine Learning

Harvard University has developed an [Introduction to Artificial Intelligence with Python](https://www.edx.org/course/cs50s-introduction-to-artificial-intelligence-with-python) course that can be audited for free.

[Artificial Intelligence: A Modern Approach](https://www.google.com/books/edition/_/koFptAEACAAJ?hl=en&sa=X&ved=2ahUKEwj3rILozs78AhV1gIQIHbMWCtsQ8fIDegQIAxBB) is the most widely used text on Artificial Intelligence in college courses.

[Mathematics for Machine Learning](https://mml-book.github.io) provides "the necessary mathematical skills to read" books that cover advanced maching learning techniques.

Grant Sanderson, also known as 3blue1brown on YouTube, has provided a very useful, high level [introduction to neural networks](https://www.3blue1brown.com/topics/neural-networks). Grant's [other videos](https://www.3blue1brown.com/#lessons) are also useful for computer and data science, and mathematics in general.

#### Deep Learning

Lightning AI's Sebastian Raschka has created a [free series on Deep Learning](https://lightning.ai/pages/courses/deep-learning-fundamentals/) and has shared his [university lectures](https://sebastianraschka.com/teaching/).

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning and has also made his his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in deep reinforcement learning.
