import os

import cv2

from helpers import *

processed_path = os.path.join(get_root_dir(), "data_processed\\")
data_path = os.path.join(get_root_dir(), "data\\")


def crop_background(path):
    target_folder = path.split("\\")[-2]
    fname = path.split("\\")[-1]
    img = cv2.imread(path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("otsu.png", thresholded)
    x, y, w, h = cv2.boundingRect(thresholded)  # bounding box
    foreground = img[y:y + h, x:x + w]
    # TODO: check that it writes the final image into the new path
    cv2.imwrite(f"{processed_path}\\{target_folder}\\{fname}", foreground)


def main():
    labels = os.listdir(data_path)
    # print(data_path)
    # print(labels)
    for label in labels:
        label_path = os.path.join(data_path, label)
        print(label_path)
        for fname in os.listdir(label_path):
            f_path = os.path.join(label_path, fname)
            print(f_path)

    # crop_background(os.path.join(data_path, "\\",labels[0]))
    pass


if __name__ == "__main__":
    main()
