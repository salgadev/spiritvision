import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import get_image_files, RandomSplitter, parent_label, Normalize
from fastai.metrics import accuracy
from fastai.vision.augment import RandomResizedCrop, aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner

# Only uncomment for debugging, uses a lot of memory
# from fastai.vision.widgets import *

from fastbook import *
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.models import resnet18, resnet34, resnet50

# from fastai
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_root_dir():
    """
    The 'scripts' folder is hardcoded since the root is expected to be its parent
    """
    current_dir = Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1] == 'scripts'][0].parent


def get_data_dir():
    return os.path.join(get_root_dir(), "data")


def get_raw_data_dir():
    return os.path.join(get_root_dir(), "data", "raw")


def get_processed_data_dir():
    return os.path.join(get_root_dir(), "data", "processed")


def get_models_dir():
    return os.path.join(get_root_dir(), "models")


def get_scripts_dir():
    return os.path.join(get_root_dir(), "scripts")


def save_to_model_folder(learner, title):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))

    fname = f'{title}-{timestamp}'
    learner.save(os.path.join(get_models_dir(), fname))
    return fname


def make_data_loader(data_path, batch_size):
    item_transforms = [Resize(size=768, method='Squish'),
                       CropPad(640, pad_mode='border'),
                       RandomResizedCrop(460)]
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(0.2),
                      get_y=parent_label,
                      item_tfms=item_transforms,
                      batch_tfms=[
                          # 70% with mult=2
                          *aug_transforms(size=224, mult=3, max_warp=0),
                          Normalize.from_stats(*imagenet_stats)])

    return block.dataloaders(data_path, bs=batch_size)


def resnet_learner(data_loader, architecture=18):
    if architecture == 18:
        return vision_learner(data_loader, resnet18, metrics=accuracy)
    elif architecture == 34:
        return vision_learner(data_loader, resnet34, metrics=accuracy)
    elif architecture == 50:
        return vision_learner(data_loader, resnet50, metrics=accuracy)
    else:
        raise RuntimeError(f'Select a compatible ResNet Architecture: 18, 34 or 50')


def save_confusion_matrix_plot(confusion_matrix, labels, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='g')

    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap="Blues")
    plt.savefig(path, dpi=120)

    print(f'Saved confusion matrix to {path}')
