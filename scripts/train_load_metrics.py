from fastai.vision.widgets import *
from sklearn.metrics import ConfusionMatrixDisplay
from helpers import *

import argparse


def main(arguments):
    arch = int(arguments.arch)

    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_data_dir(), batch_size=9)

    learn = resnet_learner(data_loader, arch)

    load_path = os.path.join(get_models_dir(), f"resnet{arch}_model")
    learn.load(load_path)

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    cm_png_path = os.path.join(get_root_dir(), "confusion_matrix.png")

    save_confusion_matrix_plot(confusion_matrix=cm, labels=data_loader.vocab, path=cm_png_path)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()
    main(args)
