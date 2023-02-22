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

As of 22 February 2023: this project will reduce vendor/provider examples in favor of a more efficient end-to-end example that focuses on concepts and methodologies instead of vendor examples. The core purpose of this repo continues to be providing [Lightning.ai](https://lightning.ai) users with a foundational example from which they can either upskill from, or use to familiarize with the Lightning ecosystem.

The following resources will be used to accomplish this:

- [Lightning](Lightning.ai): developing and serving the model
- [Optuna](https://optuna.readthedocs.io/en/stable/): hyperparameter optimization trials
- [Aim](https://aimstack.io): experiment management
- [PyneCone](https://pynecone.io): data app as a front end
- [Vercel](https://vercel.com): deploying an [exported, static PyneCone app](https://pynecone.io/docs/hosting/self-hosting)
- [Supabase](https://supabase.com): user management
- [FastAPI](https://fastapi.tiangolo.com): developing an API to serve the model with
- [Locust](https://github.com/locustio/locust): API load balance testing

Care was taken to select open source projects that may help reduce vendor lock and enable easier implementation of secure, on-prem solutions.

Changes can be tracked on [enhancement/clean-example](https://github.com/JustinGoheen/lightning-pod-example/tree/enhancement/clean-example) until that branch is merged.

A clean template for research projects is found at [lightning-pod](https://github.com/JustinGoheen/lightning-pod).
