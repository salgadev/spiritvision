__version__ = "0.1.0"

import os
from datetime import datetime
from pathlib import Path


def get_root_dir():
    """
    The 'spiritvision' folder is hardcoded since the root is expected to be its parent
    :return: path to package root (e.g. mezcal/src/spiritvision)
    """
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'spiritvision'][0].parent


def get_data_dir():
    """
    :return: path to data folder, which should be separate (one level above) the source code
    """
    return os.path.join(get_root_dir().parent, "data")


def get_raw_data_dir():
    return os.path.join(get_data_dir(), "raw")


def get_processed_data_dir():
    return os.path.join(get_data_dir(), "processed")


def get_models_dir():
    return os.path.join(get_root_dir().parent, "models")


def save_to_model_folder(learner, title):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}'
    learner.save(os.path.join(get_models_dir(), fname))
    return fname
