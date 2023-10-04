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

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
from pytorch_lightning.utilities.model_summary import ModelSummary

from dash import dash_table
from visionlab import config, LabModule


def make_metrics_summary():
    summary = json.load(open(config.Paths.wandb_summary))
    summary = dict(summary)
    collection = {
        "Training Loss": summary["training_loss"],
        "Val Loss": summary["val_loss"],
        "Val Acc": summary["val_acc"],
    }
    return collection


def create_figure(image, title_text):
    image = np.transpose(image.numpy(), (1, 2, 0))
    fig = px.imshow(image)
    fig.update_layout(
        title=dict(
            text=title_text,
            font_family="Ucityweb, sans-serif",
            font=dict(size=24),
            y=0.05,
            yanchor="bottom",
            x=0.5,
        ),
        height=300,
    )
    return fig


def make_model_layer_table(model_summary: list):
    model_layers = model_summary[:-4]
    model_layers = [i for i in model_layers if not all(j == "-" for j in i)]
    model_layers = [i.split("|") for i in model_layers]
    model_layers = [[j.strip() for j in i] for i in model_layers]
    model_layers[0][0] = "Layer"
    header = model_layers[0]
    body = model_layers[1:]
    table = pd.DataFrame(body, columns=header)
    table = dash_table.DataTable(
        data=table.to_dict("records"),
        columns=[{"name": i, "id": i} for i in table.columns],
        style_cell={
            "textAlign": "left",
            "font-family": "FreightSans, Helvetica Neue, Helvetica, Arial, sans-serif",
        },
        style_as_list_view=True,
        style_table={
            "overflow-x": "auto",
        },
        style_header={"border": "0px solid black"},
    )
    return table


def make_model_param_text(model_summary: list):
    model_params = model_summary[-4:]
    model_params = [i.split("  ") for i in model_params]
    model_params = [[i[0]] + [i[-1]] for i in model_params]
    model_params = [[j.strip() for j in i] for i in model_params]
    model_params = [i[::-1] for i in model_params]
    model_params[-1][0] = "Est. params size (MB)"
    model_params = ["".join([i[0], ": ", i[-1]]) for i in model_params]
    return model_params


def make_model_summary():
    available_trials = os.listdir(config.Paths.trials)
    available_trials.remove("README.md")
    latest_checkpoint = available_trials[0]
    chkpt_filename = os.path.join(config.Paths.trials, latest_checkpoint)
    model = LabModule.load_from_checkpoint(chkpt_filename)
    model_summary = ModelSummary(model)
    model_summary = model_summary.__str__().split("\n")
    model_layers = make_model_layer_table(model_summary)
    model_params = make_model_param_text(model_summary)
    return {"layers": model_layers, "params": model_params}


def find_index(dataset, label, label_idx):
    for i in range(len(dataset)):
        if dataset[i][label_idx] == label:
            return i
