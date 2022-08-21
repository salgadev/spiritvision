from fastai.vision.widgets import *
from helpers import *
import argparse


def main(arguments):
    arch = int(arguments.arch)

    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_processed_data_dir(), batch_size=9)

    learn = resnet_learner(data_loader, arch)
    learn.fine_tune(10)

    save_path = os.path.join(get_models_dir(), f"resnet{arch}_model")
    learn.save(save_path)

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    cm_png_path = os.path.join(get_root_dir(), "confusion_matrix.png")

    save_confusion_matrix_plot(confusion_matrix=cm, labels=data_loader.vocab, path=cm_png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()
    main(args)
