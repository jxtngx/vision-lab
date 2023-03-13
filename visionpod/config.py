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


class Settings:
    seed = 42


class Paths:
    filepath = Path(__file__)
    project = filepath.parents[1]
    package = filepath.parent
    logs = os.path.join(project, "logs")
    torch_profiler = os.path.join(logs, "torch_profiler")
    simple_profiler = os.path.join(logs, "simple_profiler")
    tensorboard = os.path.join(logs, "tensorboard")
    checkpoints = os.path.join(project, "models", "checkpoints")
    model = os.path.join(project, "models", "onnx", "model.onnx")
    predictions = os.path.join(project, "data", "predictions", "predictions.pt")
    dataset = os.path.join(project, "data", "cache")
    splits = os.path.join(project, "data", "training_split")
    train_split = os.path.join(splits, "train.pt")
    val_split = os.path.join(splits, "val.pt")
    test_split = os.path.join(splits, "test.pt")
    wandb_logs = os.path.join(project, "logs", "wandb")
    wandb_summary = os.path.join(project, "logs", "wandb", "wandb", "latest-run", "files", "wandb-summary.json")
    optuna = os.path.join(project, "logs", "optuna")


class Args:
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
    _maybe_use_mps = dict(accelerator="mps", devices=1) if MPSAccelerator.is_available() else {}
    fast_flags = dict(
        max_epochs=5,
        precision=16,
        **_maybe_use_mps,
    )
    default_flags = dict(
        max_epochs=100,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        precision=16,
        **_maybe_use_mps,
    )
    sweep_flags = dict(
        max_epochs=10,
        callbacks=[
            EarlyStopping(monitor="training_loss", mode="min"),
        ],
    )


class Sweep:
    config = dict(
        method="random",
        metric={"goal": "maximize", "name": "val_acc"},
        parameters={
            "lr": {"min": 0.0001, "max": 0.1},
            "optimizer": {"distribution": "categorical", "values": ["Adam", "RMSprop", "SGD"]},
            # "dropout": {"min": 0.2, "max": 0.5},
        },
    )


class DataModule:
    batch_size = 128
    mean = [0.49139968, 0.48215841, 0.44653091]
    stddev = [0.24703223, 0.24348513, 0.26158784]
    inverse_mean = [-i for i in mean]
    inverse_stddev = [1 / i for i in stddev]
    cifar_norm = transforms.Normalize(mean=mean, std=stddev)
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
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
    norm_test_transform = transforms.Compose([transforms.ToTensor(), cifar_norm])
    # see https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
    inverse_transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=inverse_stddev),
            transforms.Normalize(mean=inverse_mean, std=[1.0, 1.0, 1.0]),
        ]
    )
