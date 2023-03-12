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
DATASETPATH = os.path.join(PROJECTPATH, "data", "cache")
SPLITSPATH = os.path.join(PROJECTPATH, "data", "training_split")
TRAINSPLITPATH = os.path.join(PROJECTPATH, "data", "training_split", "train.pt")
VALSPLITPATH = os.path.join(PROJECTPATH, "data", "training_split", "val.pt")
TESTSPLITPATH = os.path.join(PROJECTPATH, "data", "training_split", "test.pt")
WANDBPATH = os.path.join(PROJECTPATH, "logs", "wandb")
WANDBSUMMARYPATH = os.path.join(PROJECTPATH, "logs", "wandb", "wandb", "latest-run", "files", "wandb-summary.json")
OPTUNAPATH = os.path.join(PROJECTPATH, "logs", "optuna")


# MODULE AND MODEL KWARGS
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
_maybe_use_mps = dict(accelerator="mps", devices=1) if MPSAccelerator.is_available() else {}
GLOBALSEED = 42
FASTTRAINFLAGS = dict(
    max_epochs=5,
    precision=16,
    **_maybe_use_mps,
)
TRAINFLAGS = dict(
    max_epochs=50,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    precision=16,
    **_maybe_use_mps,
)
SWEEPFLAGS = dict()

# DATAMODULE
BATCHSIZE = 128
mean = [0.49139968, 0.48215841, 0.44653091]
stddev = [0.24703223, 0.24348513, 0.26158784]
cifar_norm = transforms.Normalize(mean=mean, std=stddev)
TESTTRANSFORM = transforms.Compose([transforms.ToTensor()])
TRAINTRANSFORMS = transforms.Compose(
    [
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
    ]
)
NORMTRAINTRANSFORMS = transforms.Compose(
    [
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        cifar_norm,
    ]
)
NORMTESTTRANSFORM = transforms.Compose([transforms.ToTensor(), cifar_norm])
# see https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
INVERSETRANSFORM = transforms.Compose(
    [
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / i for i in stddev]),
        transforms.Normalize(mean=mean, std=[1.0, 1.0, 1.0]),
    ]
)
