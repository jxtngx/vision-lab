# Vision Lab

## Overview

Vision lab is a public template for artificial intelligence and machine learning research projects using Lightning AI's [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

The recommended way for Vision lab users to create new repos is with the [use this template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template) button.

## The Structure

### Source Module

`visionlab.core` should contain code for the Lightning Module and Trainer.

`visionlab.components` should contain experiment utilities grouped by purpose for cohesion.

`visionlab.pipeline` should contain code for data acquistion and preprocessing, and building a TorchDataset and LightningDataModule.

`visionlab.api` should contain code for model serving APIs built with [FastAPI](https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi).

`visionlab.cli` should contain code for the command line interface built with [Click](https://click.palletsprojects.com/en/8.1.x/) and [Rich](https://rich.readthedocs.io/en/stable/).

`visionlab.pages` should contain code for data apps built with streamlit.

`visionlab.conf` can assist with project, trainer, and sweep configurations.

### Project Root

<details>
    <summary>Root Directories and Files</summary>
    <br>

`app.py` is the Lightning App.

`assets` directory contains CSS and images for pages.

`data` directory should be used to cache the TorchDataset and training splits locally if the size of the dataset allows for local storage. additionally, this directory should be used to cache predictions during HPO sweeps.

`docs` directory should be used to store technical documentation.

`logs` directory will store logs generated from experiment managers and profilers.

`models` directory will store training checkpoints and the pre-trained production model.

`notebooks` directory can be used to present exploratory data analysis, explain math concepts, and create a presentation notebook to accompany a conference style paper.

`requirements` directory should mirror base requirements and extras found in setup.cfg. the requirements directory and _requirements.txt_ at root are required by the basic CircleCI GitHub Action.

`tests` module contains unit and integration tests targeted by pytest.

`.lightning` and `.lightningignore` are used by Lightning as config files.

`setup.py` `setup.cfg` `pyproject.toml` and `MANIFEST.ini` assist with packaging the Python project.

`.pre-commit-config.yaml` is required by pre-commit to install its git-hooks.

</details>

## Base Requirements and Extras

Vision lab installs minimal requirements out of the box, and provides extras to make creating robust virtual environments easier. To view the requirements, in [setup.cfg](setup.cfg), see `install_requires` for the base requirements and `options.extras_require` for the available extras.

The recommended install is as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```
