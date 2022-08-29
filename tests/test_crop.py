import os

import cv2
import pytest

from helpers import *
from crop import *


# TODO: test cases
# no images
# some images
# all images


def test_path_is_not_an_image():
    """Unit test to check if the path used in the get_folder_and_filename function is a .jpg, .png or .bmp file and if
    returned parent folder is valid
    """
    with pytest.raises(TypeError):
        test_path = os.path.join(get_scripts_dir(), "crop.py")
        get_folder_and_filename(test_path)
    return


def test_path_has_some_image_files():
    """
    Unit test to check if the path used in the crop_background function actually has any some image files but not all
    """
    # TODO: test calling path returning functions from helpers.py, such as
    # get_root_dir()
    # get_data_dir()
    # get_raw_data_dir()
    return


def test_path_has_all_image_files():
    """
    Unit test to check if the path used in the crop_background function actually has any no image
    """
    # TODO: test calling path returning functions from helpers.py, such as
    # get_root_dir()
    # get_data_dir()
    # get_raw_data_dir()
    return


def test_path_is_not_processed_data_dir():
    """
    Unit test to check if the path used in crop_background function is not the processed data directory, since that would probably cause circular redundancy errors or other opencv errors.
    """
    return
