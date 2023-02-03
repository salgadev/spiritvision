from fastai.interpret import ClassificationInterpretation
from helpers import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',
                        required=True,
                        help="ResNet architecture. Choose from 18, 34 or 50")

    args = parser.parse_args()
    arch = int(args.arch)

    # batch size of 9 because of small dataset
    data_loader = make_data_loader(get_processed_data_dir(),
                                   batch_size=64)

    learn = resnet_learner(data_loader, arch)
    learn.fine_tune(10)

    now = datetime.now()
    timestamp = str(now.strftime("%Y%h%d_%I%M%p"))

    save_path = os.path.join(get_models_dir(),
                             f"resnet_{arch}_{timestamp}")
    learn.save(save_path)

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()

    cm_png_path = f"{save_path}_confusion_matrix.PNG"

    save_confusion_matrix_plot(confusion_matrix=cm,
                               labels=data_loader.vocab,
                               path=cm_png_path)


if __name__ == "__main__":
    main()
