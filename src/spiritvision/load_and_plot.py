import os
import argparse

from fastai.interpret import ClassificationInterpretation

from spiritvision import get_processed_data_dir, get_models_dir, get_root_dir
from spiritvision.ai import make_data_loader, resnet_learner, save_confusion_matrix_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()

    arch = int(args.arch)

    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_processed_data_dir(), batch_size=9)

    learn = resnet_learner(data_loader, arch)

    load_path = os.path.join(get_models_dir(), f"resnet{arch}_model")
    learn.load(load_path)

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    cm_png_path = os.path.join(get_root_dir(), "confusion_matrix.png")

    save_confusion_matrix_plot(confusion_matrix=cm, labels=data_loader.vocab, path=cm_png_path)


if __name__ == "__main__":
    main()