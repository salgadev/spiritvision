import os
from fastai.interpret import ClassificationInterpretation
import argparse

import spiritvision as sv


def main(arguments):
    arch = int(arguments.arch)

    # batch size of 9 because of small dataset
    data_loader = sv.make_data_loader(sv.get_processed_data_dir(), batch_size=9)

    learn = sv.resnet_learner(data_loader, arch)
    learn.fine_tune(10)

    save_path = os.path.join(sv.get_models_dir(), f"resnet{arch}_model")
    learn.save(save_path)

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    cm_png_path = os.path.join(sv.get_root_dir(), f"confusion_matrix_ResNet{arch}.png")

    sv.save_confusion_matrix_plot(confusion_matrix=cm, labels=data_loader.vocab, path=cm_png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()
    main(args)
