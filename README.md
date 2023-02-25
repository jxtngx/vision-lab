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

# Lightning Pod Example

<a href="https://lightning.ai" ><img src ="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white" height="28"/></a> <img src ="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" height="30"/> <img src ="https://img.shields.io/badge/Supabase-181818?style=for-the-badge&logo=supabase&logoColor=white" height="30"/>

<!-- [![codecov](https://codecov.io/gh/JustinGoheen/lightning-pod-example/branch/main/graph/badge.svg)](https://codecov.io/gh/JustinGoheen/lightning-pod-example)
![CircleCI](https://circleci.com/gh/JustinGoheen/lightning-pod-example.svg?style=shield) -->

</div>

## Overview

The core purpose of this repo is to provide [Lightning.ai](https://lightning.ai) users with an end-to-end template to upskill from, or use to familiarize with the Lightning ecosystem. It uses [Lightning](https://lightning.ai), [ReactJS](https://reactjs.org) + the [Vercel](https://vercel.com) ecosystem, and [Supabase](https://supabase.com); and is built from [Lightning Pod](https://github.com/JustinGoheen/lightning-pod)'s structural concepts, with the exception that this project replaces lightning_pod.pages data apps with a ReactUI.

_As of 22 February 2023:_ this project will reduce vendor examples in favor of a more efficient end-to-end flow that focuses on concepts and methodologies instead of multiple examples for the same task.

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
- [Aim](https://aimstack.io): experiment management
- [FastAPI](https://fastapi.tiangolo.com): developing an API to serve the model with
- [Locust](https://github.com/locustio/locust): API load balance testing

**User Interfaces**

- [Supabase](https://supabase.com): database, storage, and user management
- [TurboRepo](https://turbo.build) + [NextJS](https://nextjs.org) + [React](https://reactjs.org): front end
- [Vercel](https://vercel.com): hosting the front end, and monitoring and analytics

**Secrets Management**

- [Hashicorp Vault](https://developer.hashicorp.com/vault): secret and encryption management system

**Domain**

- Topics:
  - Financial Trading
  - Market microstructure
  - Time Series Momentum
  - Time Series Analysis
  - Optimization Methods in Finance
  - Multi-agent Systems
  - Intelligent Agents
  - Recurrent and Temporal Convolutional Networks
  - Reinforcement Learning
- Data Source: [AlpacaHQ](https://alpaca.markets)
- Domain Libraries: [Nixtla](https://www.nixtla.io)

## Setup

You must have Python and NPM installed.

To setup a virtual development environment, in terminal, do:

```sh
python3 -m venv .venv/
source .venv/bin/activate
pip install -e ".[full]"
pre-commit install
deactivate
cd lpx-ui
yarn install
cd ..
```

## AI/ML and Software Engineering Learning Resources

### Reviewing Source Code

The following three videos were created by Lightning's Thomas Chaton; the videos are extremely helpful in learning how to use code search features in VS Code to navigate a project's source code, enabling a deeper understanding of what is going on under the hood of someone else's code.

> these videos were created before PyTorch Lightning was moved into the Lightning Framework mono repo

[Lightning Codebase Deep Dive 1](https://youtu.be/aEeh9ucKUkU) <br>
[Lightning Codebase Deep Dive 2](https://youtu.be/NEpRYqdsm54) <br>
[Lightning Codebase Deep Dive 3](https://youtu.be/x4d4RDNJaZk)

### General Engineering and Tools

Lightning's founder, and their lead educator have created a series of short videos called [Lightning Bits](https://lightning.ai/pages/ai-education/#bits) for beginners who need guides for using IDEs, git, and terminal.

A long standing Python community resource has been [The Hitchhiker's Guide to Python](https://docs.python-guide.org). The "guide exists to provide both novice and expert Python developers a best practice handbook for the installation, configuration, and usage of Python on a daily basis".

[VS Code](https://code.visualstudio.com/docs) and [PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) IDEs have each provided great docs for their users. My preference is VS Code - though PyCharm does have its benefits and is absolutely a suitable alternative to VS Code. I especially like VS Code's [integrations for PyTorch and tensorboard](https://code.visualstudio.com/docs/datascience/pytorch-support). I pair [Gitkraken](https://www.gitkraken.com) and [GitLens](https://www.gitkraken.com/gitlens) with VS Code to manage my version control and contributions.

### Data Analysis

Wes McKinney, creator of Pandas and founder of Voltron Data (responsible for Ibis, Apache Arrow etc) has released his third edition of [Python for Data Analysis](https://wesmckinney.com/book/) in an open access format.

### Intro to Artificial Intelligence and Mathematics for Machine Learning

Harvard University has developed an [Introduction to Artificial Intelligence with Python](https://www.edx.org/course/cs50s-introduction-to-artificial-intelligence-with-python) course that can be audited for free.

[Artificial Intelligence: A Modern Approach](https://www.google.com/books/edition/_/koFptAEACAAJ?hl=en&sa=X&ved=2ahUKEwj3rILozs78AhV1gIQIHbMWCtsQ8fIDegQIAxBB) is the most widely used text on Artificial Intelligence in college courses.

[Mathematics for Machine Learning](https://mml-book.github.io) provides "the necessary mathematical skills to read" books that cover advanced maching learning techniques.

Grant Sanderson, also known as 3blue1brown on YouTube, has provided a very useful, high level [introduction to neural networks](https://www.3blue1brown.com/topics/neural-networks). Grant's [other videos](https://www.3blue1brown.com/#lessons) are also useful for computer and data science, and mathematics in general.

### Deep Learning

Lightning AI's Sebastian Raschka has created a [free series on Deep Learning](https://lightning.ai/pages/courses/deep-learning-fundamentals/).

NYU's Alfredo Canziani has created a [YouTube Series](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) for his lectures on deep learning. Additionally, Professor Canziani was kind enough to make his course materials public [on GitHub](https://github.com/Atcold/NYU-DLSP21).

The book [Dive into Deep Learning](http://d2l.ai/#), created by a team of Amazon engineers, is availlable for free.

DeepMind has shared several lectures series created for UCL [on YouTube](https://www.youtube.com/c/DeepMind/playlists?view=50&sort=dd&shelf_id=9).

OpenAI has created [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), an introductory series in deep reinforcement learning.

### ML Ops

Weights and Biases has created a free [ML Ops](https://www.wandb.courses/courses/effective-mlops-model-development) course.
