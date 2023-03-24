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

import torchvision
from lightning import LightningApp

from visionpod import config
from visionpod.components import TrainerFlow

os.environ["WANDB_CONFIG_DIR"] = config.ExperimentManager.WANDB_CONFIG_DIR

torchvision.disable_beta_transforms_warning()

# TODO give a really verbose example of payload
sweep_payload = dict(
    project_name="visionpod",  # the wandb project name
    trial_count=2,  # low trial count for proof of concept (POC)
    machine="default",  # 1 cpu: 0.2 USD per hour
    idle_timeout=60,  # wandb needs time to finish logging sweep
    interruptible=False,  # set to True for spot instances. False because not supported yet
    trainer_init_flags=config.Sweep.fast_trainer_flags,  # sets low max epochs for POC
    wandb_save_dir=config.Paths.wandb_logs,  # where wandb will push logs to locally
    model_kwargs=config.Module.model_kwargs,  # args required by ViT
)

# TODO give a really verbose example of payload
trainer_payload = dict(
    tune=True,  # let trainer know to expect a tuned config payload
    machine="default",  # 1 cpu: 0.2 USD per hour
    idle_timeout=30,  # give wandb time to finish
    interruptible=False,  # set to True for spot instances. False because not supported yet
    trainer_flags=config.Trainer.fast_flags,  # sets low max epochs for POC
    model_kwargs=config.Module.model_kwargs,  # args required by ViT
)

# TODO figure out why app is not terminating locally
app = LightningApp(TrainerFlow(sweep_payload=sweep_payload, trainer_payload=trainer_payload))
