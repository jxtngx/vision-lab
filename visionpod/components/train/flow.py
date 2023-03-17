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


from lightning import LightningApp, LightningFlow

from visionpod import config
from visionpod.components.sweep.work import SweepWork
from visionpod.components.train.work import TrainerWork


class TrainerFlow(LightningFlow):
    def __init__(self):
        super().__init__()

        self._sweep_work = SweepWork(
            trainer_init_flags=config.Sweep.fast_trainer_flags,
            **config.Sweep.fast_init_kwargs,
        )

        self._trainer_work = TrainerWork(
            fast_train_run=True,
            sweep=False,
            trainer_flags=config.Trainer.fast_flags,
        )

    def run(self):
        # should be blocking
        self._sweep_work.run()
        # stop after optimization is complete
        self._sweep_work.stop()

        # also blocking
        self._trainer_work.run()
        # stop after training is complete
        self._trainer_work.stop()


app = LightningApp(TrainerFlow())
