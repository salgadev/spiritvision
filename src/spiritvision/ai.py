from fastbook import *
from fastai.vision.widgets import *

from fastai.metrics import accuracy
from fastai.vision.learner import vision_learner

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


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
    """
    :param data_loader: output from make_data_loader function
    :param architecture: ResNet18, 34, 50. ResNet34 is set by default if blank
    :return: vision_learner AKA fastai learner object
    """
    print(f"Making a RestNet{architecture} Learner")
    if architecture == 18:
        return vision_learner(data_loader, models.resnet18, metrics=accuracy)
    elif architecture == 34:
        return vision_learner(data_loader, models.resnet34, metrics=accuracy)
    elif architecture == 50:
        return vision_learner(data_loader, models.resnet50, metrics=accuracy)
    else:
        raise RuntimeError(f"Arch should only be 18, 34 or 50 for ResNet learners. You entered: {architecture}")


def make_data_loader(data_path, batch_size):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(0.2),
                      get_y=parent_label,
                      item_tfms=RandomResizedCrop(460),
                      batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)])

    return block.dataloaders(data_path, bs=batch_size)
