
import os
import re

import cv2

from scripts.helpers import get_data_dir, get_raw_data_dir, get_processed_data_dir
import numpy as np
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
        return filename


def crop_background(path):
    target_folder = path.split('\\')[-2]
    fname = path.split('\\')[-1]

    label_folder = os.path.join(processed_data_path, target_folder)
    background_folder = os.path.join(processed_data_path, 'background')
    for folder in [label_folder, background_folder]:
        os.makedirs(folder, exist_ok=True)

    img = cv2.imread(path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the image and allow user to select background
    print("Please select the background by clicking and dragging around the area.")
    x, y, w, h = cv2.selectROI(img, False)

    if w == 0 or h == 0:
        print(f"Skipping {path} as the bounding box has 0 width or height")
        cv2.destroyAllWindows()
        return

    cv2.destroyAllWindows()
    selected_region = grayscale[y:y + h, x:x + w]

    threshold = np.mean(selected_region)

    # _, thresholded = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Crop the foreground object
    foreground = img[y:y + h, x:x + w]
    # Get the background by subtracting the foreground from the original image
    background_image = img.copy()
    background_image[np.where(thresholded == 255)] = 0

    class_path = os.path.join(label_folder, f"cropped-{fname}")
    background_path = os.path.join(background_folder, f"background-{fname}")

    for saving_path, image in {class_path: foreground,
                               background_path: background_image}.items():
        if not os.path.isfile(saving_path):
            print(f"{saving_path} Saved!")
            cv2.imwrite(saving_path, image)
        else:
            print(f"{saving_path} already exists!")


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
