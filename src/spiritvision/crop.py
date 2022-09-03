import os

import cv2
from helpers import *

processed_path = get_processed_data_dir()
data_path = get_raw_data_dir()


def get_folder_and_filename(path):
    """Splits image path into its filename and its parent folder before processing
    :param: path: must be the path to an image
    :return: filename and the path to parent folder
    """
    target_folder = path.split("\\")[-2]
    if not os.path.isdir(target_folder):
        raise TypeError(f"{target_folder} is not a valid folder.")

    fname = path.split("\\")[-1]
    f, extension = fname.lower().split(".")
    if extension not in ["png", "bmp", "jpg", "jpeg"]:
        raise TypeError(f"{fname} has a non-image extension: {extension}.")

    return fname, target_folder


def crop_background(path, debug=False):
    # TODO: if debug is True should not delete ostu.png, otherwise delete it always
    filename, target_folder = get_folder_and_filename(path)
    img = cv2.imread(path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("otsu.png", thresholded)
    x, y, w, h = cv2.boundingRect(thresholded)  # bounding box
    foreground = img[y:y + h, x:x + w]

    label_folder = f"{processed_path}\\{target_folder}"
    if not os.path.isdir(label_folder):
        os.mkdir(label_folder)

    target_path = f"{label_folder}\\cropped-{filename}"
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
