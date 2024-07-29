
import os
from pathlib import Path
# Import the pandas package
import pandas as pd
import torch
from cjm_pytorch_utils.core import set_seed, pil_to_tensor, tensor_to_pil, get_torch_device, denorm_img_tensor, move_data_to_device


# Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
seed = 123
set_seed(seed)

# Setting the Device and Data Type
device = get_torch_device()
dtype = torch.float32
# device, dtype

# Setting the Directory Paths
# The name for the project
project_name = f"pytorch-keypoint-r-cnn"

# The path for the project folder
project_dir = Path(f"./{project_name}/")

# Create the project directory if it does not already exist
project_dir.mkdir(parents=True, exist_ok=True)

# Define path to store datasets
dataset_dir = Path("./Datasets/")
# Create the dataset directory if it does not exist
dataset_dir.mkdir(parents=True, exist_ok=True)

# Define path to store archive files
archive_dir = dataset_dir/'../Archive'
# Create the archive directory if it does not exist
archive_dir.mkdir(parents=True, exist_ok=True)

# Creating a Series with the paths and converting it to a DataFrame for display
s = pd.Series({
    "Project Directory:": project_dir,
    "Dataset Directory:": dataset_dir,
    "Archive Directory:": archive_dir
}).to_frame().style.hide(axis='columns')

print(s)
