from datetime import datetime
from pathlib import Path


def get_root_dir():
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'mezcal'][0]


def save_model(learner, title):
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}'
    learner.save(fname)
    return fname