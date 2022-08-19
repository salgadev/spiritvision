from fastai.vision.widgets import *
from helpers import *


def main():
    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_data_dir(), batch_size=9)

    arch = 34
    learn = resnet_learner(data_loader, arch)
    learn.fine_tune(10)

    save_to_model_folder(learn, f"resnet{arch}_model.pth")


if __name__ == "__main__":
    main()
