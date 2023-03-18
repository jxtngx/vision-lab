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


from typing import Any, Dict

from lightning import LightningFlow

from visionpod.components.sweep.work import SweepWork
from visionpod.components.train.work import TrainerWork


class TrainerFlow(LightningFlow):
    def __init__(self, sweep_payload: Dict[str, Any], trainer_payload: Dict[str, Any]):
        super().__init__()
        self.sweep_work = SweepWork(**sweep_payload)
        self.trainer_work = TrainerWork(**trainer_payload)

    def run(self):
        self.sweep_work.run()
        self.trainer_work.run(sweep_id=self.sweep_work.sweep_id)
        self.stop()
