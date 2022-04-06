import fastbook
import fastai
from fastai.vision.widgets import *
from fastbook import *
import zipfile
import os
import datetime
from helpers import *

# TOO SLOW ON PYCHARM FOR SOME REASON


def main():
    path = Path(get_data_dir())
    files = get_image_files(path)
    print(files)

    mezcal = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_items=get_image_files,
                       splitter=RandomSplitter(0.2),
                       get_y=parent_label,
                       item_tfms=RandomResizedCrop(460),
                       batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)])

    # batch size of 9 because of small dataset
    dls = mezcal.dataloaders(path, bs=9)
    print(f"The classes are: {dls.vocab}")

    learn = cnn_learner(dls, resnet34, pretrained=True, metrics=error_rate).to_fp16()
    learn.fine_tune(10)
    model = save_model(learn, "apr5")

    pass


if __name__ == "__main__":
    main()
