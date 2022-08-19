import os
from datetime import datetime
from pathlib import Path

from fastai.metrics import accuracy
from fastai.vision import models
from fastai.vision.learner import vision_learner

from fastai.vision.widgets import *
from fastbook import *


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


def save_to_model_folder(learner, title):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}'
    learner.save(os.path.join(get_models_dir(), fname))
    return fname


def get_resnet18_recipe():
    return os.path.join(get_root_dir(), "recipes", "resnet-18-original.md")


def make_data_loader(data_path, batch_size):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(0.2),
                      get_y=parent_label,
                      item_tfms=RandomResizedCrop(460),
                      batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)])

    return block.dataloaders(data_path, bs=batch_size)


def resnet_learner(data_loader, architecture=34):
    if architecture == 34:
        return vision_learner(data_loader, models.resnet34, metrics=accuracy)
    elif architecture == 50:
        return vision_learner(data_loader, models.resnet50, metrics=accuracy)
    else:
        return vision_learner(data_loader, models.resnet18, metrics=accuracy)
