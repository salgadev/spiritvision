import os
import json

from scripts.helpers import get_processed_data_dir

# Define the path to the dataset
dataset_path = get_processed_data_dir()

# Get a list of the class names from the dataset
class_names = os.listdir(dataset_path)

# Create a dictionary that maps class number to class name
index_to_name = {str(i): class_names[i] for i in range(len(class_names))}

# Save the dictionary as a JSON file
with open('../index_to_name.json', 'w') as f:
    json.dump(index_to_name, f)
