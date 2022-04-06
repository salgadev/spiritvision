import os
from datetime import datetime
from pathlib import Path


def get_root_dir():
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'mezcal'][0]


def get_data_dir():
    return os.path.join(get_root_dir(), "data")


def save_model(learner, title):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}'
    learner.save(fname)
    return fname
