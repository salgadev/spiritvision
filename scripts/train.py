import fastai
from fastai.vision.widgets import *
from fastbook import *
import zipfile
import os
import datetime
from helpers import *


def main():
    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_data_dir(), batch_size=9)

    arch = 34
    learn = resnet_learner(data_loader, arch)
    learn.fine_tune(10)

    save_to_model_folder(learn, f"resnet{arch}_model.pth")


def resnet_learner(data_loader, architecture=34):
    if architecture == 34:
        return vision_learner(data_loader, models.resnet34, metrics=accuracy)
    elif architecture == 50:
        return vision_learner(data_loader, models.resnet50, metrics=accuracy)
    else:
        return vision_learner(data_loader, models.resnet18, metrics=accuracy)


if __name__ == "__main__":
    main()
