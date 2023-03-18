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

from lightning import LightningApp

from visionpod import config
from visionpod.components import TrainerFlow

sweep_payload = dict(
    trainer_init_flags=config.Sweep.fast_trainer_flags,
    wandb_save_dir=config.Paths.wandb_logs,
    project_name="visionpod",
    trial_count=2,
    parallel=False,
)

trainer_payload = dict(
    trainer_flags=config.Trainer.fast_flags,
    model_kwargs=config.Module.model_kwargs,
    tune=True,
)

app = LightningApp(TrainerFlow(sweep_payload=sweep_payload, trainer_payload=trainer_payload))
