import matplotlib as plt
from fastbook import DataBlock, CategoryBlock

from fastai.data.transforms import get_image_files, RandomSplitter, parent_label, Normalize
from fastai.metrics import accuracy
from fastai.vision import models
from fastai.vision.augment import RandomResizedCrop, aug_transforms
from fastai.vision.core import imagenet_stats
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from sklearn.metrics import ConfusionMatrixDisplay


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


def resnet_learner(data_loader, architecture=34):
    if architecture == 34:
        return vision_learner(data_loader, models.resnet34, metrics=accuracy)
    elif architecture == 50:
        return vision_learner(data_loader, models.resnet50, metrics=accuracy)
    else:
        return vision_learner(data_loader, models.resnet18, metrics=accuracy)


def make_data_loader(data_path, batch_size):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(0.2),
                      get_y=parent_label,
                      item_tfms=RandomResizedCrop(460),
                      batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)])

    return block.dataloaders(data_path, bs=batch_size)
