import os
import shutil
import splitfolders
from scripts.helpers import get_processed_data_dir, get_data_dir


def rename_data_files(dataset_path, target_path):
    # Iterate through all subfolders and files in the dataset path
    for root, _, files in os.walk(dataset_path):
        for index, file_name in enumerate(files, start=1):
            # Create the new file name with the index
            _, extension = os.path.splitext(file_name)
            new_file_name = f"{index:04}{extension}"  # 0001.jpg, 0002.jpg, ...

            # Create the new target folder path by replacing the dataset path
            # with the target path and preserving the folder structure
            relative_path = os.path.relpath(root, dataset_path)
            new_folder_path = os.path.join(target_path, relative_path)

            # Create the new target folder if it doesn't exist
            os.makedirs(new_folder_path, exist_ok=True)

            # Copy the file to the new target folder with the new name
            old_file_path = os.path.join(root, file_name)
            new_file_path = os.path.join(new_folder_path, new_file_name)

            # Check if the file already exists in the target folder
            if not os.path.exists(new_file_path):
                shutil.move(old_file_path, new_file_path)


def clean_up(path):
    # Recursively iterate through all subfolders and files in the given path
    for root, dirs, files in os.walk(path, topdown=False):
        # Delete empty subfolders
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

        # Delete empty root folder if it is empty after deleting subfolders
        if not os.listdir(root):
            os.rmdir(root)


if __name__ == "__main__":
    # Example usage:
    processed_data_path = get_processed_data_dir()
    release_path = os.path.join(get_data_dir(), 'release')

    # Split the dataset into train and test folders using splitfolders
    splitfolders.ratio(processed_data_path, output=release_path, seed=42, ratio=(0.80, 0.15, 0.05), group_prefix=None)

    # Rename the files in the train and test folders
    for folder_name in ['train', 'val', 'test']:
        folder_path = os.path.join(release_path, folder_name)
        rename_data_files(folder_path, folder_path)

    # Clean up empty folders after moving the data files
    clean_up(release_path)
