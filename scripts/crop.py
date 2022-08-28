import os

import cv2
from helpers import *

processed_path = get_processed_data_dir()
data_path = get_raw_data_dir()


def crop_background(path):
    target_folder = path.split("\\")[-2]
    fname = path.split("\\")[-1]
    img = cv2.imread(path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("otsu.png", thresholded)
    x, y, w, h = cv2.boundingRect(thresholded)  # bounding box
    foreground = img[y:y + h, x:x + w]

    label_folder = f"{processed_path}\\{target_folder}"
    if not os.path.isdir(label_folder):
        os.mkdir(label_folder)

    target_path = f"{label_folder}\\cropped-{fname}"
    if not os.path.isfile(target_path):
        print(f"Saving {target_path}")
        cv2.imwrite(target_path, foreground)
    else:
        print(f"{target_path} already exists!")


def main():
    labels = os.listdir(data_path)

    for label in labels:
        label_path = os.path.join(data_path, label)

        for fname in os.listdir(label_path):
            crop_background(os.path.join(label_path, fname))

    pass


if __name__ == "__main__":
    main()
