
import os
import re
import numpy as np
import cv2

from scripts.helpers import get_data_dir, get_raw_data_dir, get_processed_data_dir, get_interim_data_dir

data_path = get_data_dir()
raw_data_path = get_raw_data_dir()
processed_data_path = get_processed_data_dir()
interim_data_path = get_interim_data_dir()


def check_utf8(filename):
    pattern = re.compile(r"[^\x00-\x7F]+")
    match = pattern.search(filename)
    if match:
        print(f"The string contains non-UTF-8 characters: {match.group()}")
        new_filename = re.sub(r"[^\x00-\x7F]+", "", filename)
        return new_filename
    else:
        return filename


def save_to_interim_data_dir(filename, image_instance):
    image_saving_path = os.path.join(interim_data_path, filename)
    os.makedirs(interim_data_path, exist_ok=True)
    cv2.imwrite(image_saving_path, image_instance)


def crop_background(path):
    target_folder = path.split('\\')[-2]
    fname = path.split('\\')[-1]

    label_folder = os.path.join(processed_data_path, target_folder)
    background_folder = os.path.join(processed_data_path, 'background')
    for folder in [label_folder, background_folder]:
        os.makedirs(folder, exist_ok=True)

    img = cv2.imread(path)

    # Get image median
    median_intensity = np.median(img)

    # add median blur to image
    blurred = cv2.medianBlur(img, 7)
    save_to_interim_data_dir(f"median_blur_{fname}", blurred)

    # Convert the image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    save_to_interim_data_dir(f"edges_{fname}", edges)

    # Find the coordinates of the edges
    (rows, cols) = np.where(edges == 255)

    # Get the minimum and maximum x and y coordinates
    x_min, x_max = min(cols), max(cols)
    y_min, y_max = min(rows), max(rows)

    # Crop the image
    foreground = img[y_min:y_max, x_min:x_max]
    background = np.full(img.shape, median_intensity, dtype=np.uint8)
    background[:y_min, :] = img[:y_min, :]
    background[y_max:, :] = img[y_max:, :]
    background[:, :x_min] = img[:, :x_min]
    background[:, x_max:] = img[:, x_max:]

    class_path = os.path.join(label_folder, f"cropped-{fname}")
    background_path = os.path.join(background_folder, f"background-{fname}")

    for saving_path, image in {class_path: foreground,
                               background_path: background}.items():
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
