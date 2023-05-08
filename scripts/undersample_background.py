import os
import random
import statistics

from scripts.helpers import get_background_data_dir, get_processed_data_dir


def undersample_class(class_path):
    classes = os.listdir(get_processed_data_dir())
    # set the random seed for reproducibility
    random.seed(42)

    # determine the target number of samples for the background class
    dataset_path = os.path.dirname(class_path)
    counts = [len(os.listdir(os.path.join(dataset_path, c))) for c in classes if c != "background"]
    target_count = int(statistics.median(counts))

    # delete the excess files from the background class
    background_files = os.listdir(class_path)
    random.shuffle(background_files)
    for i in range(target_count, len(background_files)):
        os.remove(os.path.join(class_path, background_files[i]))

    # verify that each class has the same number of samples
    for c in classes:
        count = len(os.listdir(os.path.join(dataset_path, c)))
        print(f"{c}: {count} samples")


if __name__ == "__main__":
    folder_path = get_background_data_dir()
    undersample_class(folder_path)
