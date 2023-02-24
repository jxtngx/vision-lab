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

<a 
href="https://lightning.ai" ><img src ="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" height="28"/> </a>

</div>

## Overview

_As of 22 February 2023:_ this project will reduce vendor examples in favor of a more efficient end-to-end flow that focuses on concepts and methodologies instead of multiple examples for the same task. The core purpose of this repo is to provide [Lightning.ai](https://lightning.ai) users with a foundational template to upskill from, or use to familiarize with the Lightning ecosystem.

**Languages**

- [Python](https://www.python.org): data engineering, and machine learning
- [TypeScript](https://www.typescriptlang.org) and [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript): frontend design and visualization, and select backend services
- [Rust](https://www.rust-lang.org): exposure via project tooling ([ruff](https://beta.ruff.rs/docs/), [turbo](https://turbo.build))

**Data Engineering and Machine Learning**

- [Lightning](Lightning.ai): developing and serving the model
- [Optuna](https://optuna.readthedocs.io/en/stable/): hyperparameter optimization trials
- [Aim](https://aimstack.io): experiment management
- [FastAPI](https://fastapi.tiangolo.com): developing an API to serve the model with
- [Locust](https://github.com/locustio/locust): API load balance testing

**Web App**

- [Supabase](https://supabase.com) + [Postgres](https://supabase.com/docs/guides/database/overview) + [Prisma](https://supabase.com/docs/guides/integrations/prisma): database, storage, and user auth management
- [TurboRepo](https://turbo.build) + [NextJS](https://nextjs.org) + [React](https://reactjs.org): front end
- [Vercel](https://vercel.com): deploying the front end, monitoring and analytics

## Notes

Changes can be tracked on [enhancement/clean-example](https://github.com/JustinGoheen/lightning-pod-example/tree/enhancement/clean-example) until that branch is merged.

A clean template for research projects is found at [lightning-pod](https://github.com/JustinGoheen/lightning-pod).

## Comments

_Why use React and TypeScript instead of Plotly Dash, Streamlit, or Pynecone?_

Simply put, Dash and Streamlit seem more focused on enterprise clients - making use difficult for lone engineers trying to go from -1 to 0 with regard to conveying technology via a visual interface.

Pynecone is promising given it compiles to NextJS - making project deployable to Vercel; however, it is simply too new to adopt.

Learning and using TypeScript with React enables creating better frontends with tools like [material-ui](https://mui.com), MUI's [figma design kit](https://mui.com/store/items/figma-react/), and [d3js](https://d3js.org). We can learn from an applied approach, developing a single page web app as an [mvp](https://en.wikipedia.org/wiki/Minimum_viable_product)/[pre-alpha](https://en.wikipedia.org/wiki/Software_release_life_cycle).

_Why NextJS and Supabase?_

NextJS has an easy to use [developer onboarding resource](https://nextjs.org/learn/foundations/about-nextjs?utm_source=next-site&utm_medium=nav-cta&utm_campaign=next-website).

Supabase has an integration with [Vercel.](https://supabase.com/docs/guides/integrations/vercel), the creators of NextJS. Making it an easy choice.

_Why an end-to-end example?_

Going through end-to-end example will increase our value-add to cross-functional teams, as we will be more literate of concepts across the [product development](https://asana.com/resources/product-development-process), and [Dev](https://resources.github.com/devops/) + [ML Ops](https://blogs.nvidia.com/blog/2020/09/03/what-is-mlops/) life cycles.
