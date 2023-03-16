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
from visionpod.components import TrainerWork

app = LightningApp(
    TrainerWork(
        trainer_flags=config.Trainer.fast_flags,
        fast_train_run=True,
        sweep=False,
    )
)

if not config.System.is_cloud_run:
    if app.named_works[0][1].has_succeeded:
        app.named_works[0][1].stop()
