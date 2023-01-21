
import os
import re

import cv2

from scripts.helpers import get_data_dir, get_raw_data_dir, get_processed_data_dir

data_path = get_data_dir()
raw_data_path = get_raw_data_dir()
processed_data_path = get_processed_data_dir()


def check_utf8(filename):
    pattern = re.compile(r"[^\x00-\x7F]+")
    match = pattern.search(filename)
    if match:
        print(f"The string contains non-UTF-8 characters: {match.group()}")
        new_filename = re.sub(r"[^\x00-\x7F]+", "", filename)
        return new_filename

    else:
        print(f"The string is in UTF-8 encoding.")
        return filename


def crop_background(path):
    target_folder = path.split("\\")[-2]
    fname = path.split("\\")[-1]

    img = cv2.imread(path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("otsu.png", thresholded)
    x, y, w, h = cv2.boundingRect(thresholded)  # bounding box
    foreground = img[y : y + h, x : x + w]

    label_folder = f"{processed_data_path}\\{target_folder}"
    os.makedirs(label_folder, exist_ok=True)

    target_path = f"{label_folder}\\cropped-{fname}"
    if not os.path.isfile(target_path):
        print(f"Saving {target_path}")
        cv2.imwrite(target_path, foreground)
    else:
        print(f"{target_path} already exists!")


def main():
    labels = os.listdir(raw_data_path)

    for label in labels:
        label_path = os.path.join(raw_data_path, label)

        for fname in os.listdir(label_path):
            original_filepath = os.path.join(label_path, fname)
            utf8_fname = check_utf8(fname)

            if utf8_fname is not fname:
                renamed_filepath = os.path.join(label_path, utf8_fname)
                os.rename(original_filepath, renamed_filepath)
                print(f"File {original_filepath} was renamed to {renamed_filepath}")
                crop_background(renamed_filepath)
            else:
                crop_background(original_filepath)


if __name__ == "__main__":
    main()
