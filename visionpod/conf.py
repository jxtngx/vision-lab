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
from functools import partial
from pathlib import Path

import torch
from lightning.pytorch.accelerators.mps import MPSAccelerator
from lightning.pytorch.callbacks import EarlyStopping
from torchvision import transforms

# PATHS
filepath = Path(__file__)
PROJECTPATH = filepath.parents[1]
PKGPATH = filepath.parent
LOGSPATH = os.path.join(PROJECTPATH, "logs")
TORCHPROFILERPATH = os.path.join(LOGSPATH, "torch_profiler")
SIMPLEPROFILERPATH = os.path.join(LOGSPATH, "simple_profiler")
TENSORBOARDPATH = os.path.join(LOGSPATH, "tensorboard")
CHKPTSPATH = os.path.join(PROJECTPATH, "models", "checkpoints")
MODELPATH = os.path.join(PROJECTPATH, "models", "onnx", "model.onnx")
PREDSPATH = os.path.join(PROJECTPATH, "data", "predictions", "predictions.pt")
VALPATH = os.path.join(PROJECTPATH, "data", "training_split", "val.pt")
DATASETPATH = os.path.join(PROJECTPATH, "data", "cache")
SPLITSPATH = os.path.join(PROJECTPATH, "data", "training_split")
WANDBPATH = os.path.join(PROJECTPATH, "logs", "wandb")
WANDBSUMMARYPATH = os.path.join(PROJECTPATH, "logs", "wandb", "wandb", "latest-run", "files", "wandb-summary.json")
OPTUNAPATH = os.path.join(PROJECTPATH, "logs", "optuna")


# MODULE AND MODEL KWARGS
IMAGESIZE = 32
NUMCLASSES = 10
MODULEKWARGS = dict(
    lr=1e-3,
    optimizer="Adam",
)
MODELKWARGS = dict(
    image_size=32,
    num_classes=10,
    progress=False,
    weights=False,
)
MODELHYPERS = dict(
    dropout=0.25,
    attention_dropout=0.25,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    conv_stem_configs=None,
)

# TRAINER FLAGS
maybe_use_mps = dict(accelerator="mps", devices=1) if MPSAccelerator.is_available() else {}
GLOBALSEED = 42
TRAINFLAGS = dict(
    max_epochs=100,
    callbacks=[EarlyStopping(monitor="training_loss", mode="min")],
    precision=16,
    **maybe_use_mps,
)
SWEEPFLAGS = dict()

# PIPELINE
TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
    ]
)
AUTOAUGMENT = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    ]
)
