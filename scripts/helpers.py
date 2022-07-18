import os
from datetime import datetime
from pathlib import Path


def get_root_dir():
    """
    The 'scripts' folder is hardcoded since the root is expected to be its parent
    """
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'scripts'][0].parent


def get_data_dir():
    return os.path.join(get_root_dir(), "data")


def get_models_dir():
    return os.path.join(get_root_dir(), "models")


def save_model(learner, title):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}'
    learner.save(os.path.join(get_models_dir(), fname))
    return fname


def get_resnet18_recipe():
    return os.path.join(get_root_dir(), "recipes", "resnet-18-original.md")
