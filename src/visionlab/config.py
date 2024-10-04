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
import sys
from functools import partial
from pathlib import Path

import torch
from lightning.pytorch.accelerators.mps import MPSAccelerator
from lightning.pytorch.callbacks import EarlyStopping
from torchvision import transforms


class Settings:
    mps_available = MPSAccelerator.is_available()
    seed = 42
    projectname = "visionlab"
    data_version = "0"
    maybe_use_mps = dict(accelerator="mps", devices=1) if MPSAccelerator.is_available() else {}
    precision_dtype = "16-mixed" if mps_available else "32-true"
    platform = sys.platform


class Paths:
    filepath = Path(__file__)
    project = filepath.parents[2]
    package = filepath.parent
    # logs
    logs = os.path.join(project, "logs")
    torch_profiler = os.path.join(logs, "torch_profiler")
    simple_profiler = os.path.join(logs, "simple_profiler")
    tuned_configs = os.path.join(logs, "tuned_configs")
    # models
    ckpts = os.path.join(project, "checkpoints")
    model = os.path.join(project, "checkpoints", "onnx", "model.onnx")
    predictions = os.path.join(project, "data", "predictions", "predictions.pt")
    # data
    dataset = os.path.join(project, "data", "cache")
    splits = os.path.join(project, "data", "training_split")
    train_split = os.path.join(splits, f"v{Settings.data_version}-train.pt")
    val_split = os.path.join(splits, f"v{Settings.data_version}-val.pt")
    test_split = os.path.join(splits, f"v{Settings.data_version}-test.pt")


class Module:
    module_kwargs = dict(
        lr=1e-3,
        optimizer="Adam",
    )
    model_kwargs = dict(
        image_size=32,
        num_classes=10,
        progress=False,
        weights=False,
    )
    model_hyperameters = dict(
        dropout=0.25,
        attention_dropout=0.25,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        conv_stem_configs=None,
    )


class Trainer:
    train_flags = dict(
        max_epochs=100,
        precision=Settings.precision_dtype,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        **Settings.maybe_use_mps,
    )
    fast_flags = dict(
        max_epochs=2,
        precision=Settings.precision_dtype,
        **Settings.maybe_use_mps,
    )


class DataModule:
    batch_size = 128
    mean = [0.49139968, 0.48215841, 0.44653091]
    stddev = [0.24703223, 0.24348513, 0.26158784]
    inverse_mean = [-i for i in mean]
    inverse_stddev = [1 / i for i in stddev]
    cifar_norm = transforms.Normalize(mean=mean, std=stddev)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ]
    )
    norm_train_transform = transforms.Compose(
        [
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            cifar_norm,
        ]
    )
    norm_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar_norm,
        ]
    )
    # see https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
    inverse_transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=inverse_stddev),
            transforms.Normalize(mean=inverse_mean, std=[1.0, 1.0, 1.0]),
        ]
    )

