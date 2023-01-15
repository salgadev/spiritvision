"""
Convert learner object into pytorch state dictionary
"""
from fastai.vision.widgets import *
from helpers import *

import argparse


def main(arguments):
    arch = int(arguments.arch)

    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_data_dir(), batch_size=9)

    learn = get_resnet_model(data_loader, arch)

    load_path = os.path.join(get_models_dir(), f"resnet{arch}_model")
    learn.load(load_path)

    model = learn.model

    model.eval()

    # Save the state dictionary to a file
    torch.save(model.state_dict(), f'{load_path}_state_dict.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()
    main(args)
