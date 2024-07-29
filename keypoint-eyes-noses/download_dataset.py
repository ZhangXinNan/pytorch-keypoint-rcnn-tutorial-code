
from pathlib import Path
from setting_up import archive_dir, dataset_dir
from cjm_psl_utils.core import download_file, file_extract
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
import pandas as pd
from tqdm.auto import tqdm

# Loading and Exploring the Dataset
# Setting the Dataset Path
# Set the name of the dataset
dataset_name = 'labelme-keypoint-eyes-noses-dataset'

# Construct the HuggingFace Hub dataset name by combining the username and dataset name
hf_dataset = f'cj-mills/{dataset_name}'

# Create the path to the zip file that contains the dataset
archive_path = Path(f'{archive_dir}/{dataset_name}.zip')

# Create the path to the directory where the dataset will be extracted
dataset_path = Path(f'{dataset_dir}/{dataset_name}')

# Creating a Series with the dataset name and paths and converting it to a DataFrame for display
pd.Series({
    "HuggingFace Dataset:": hf_dataset,
    "Archive Path:": archive_path,
    "Dataset Path:": dataset_path
}).to_frame().style.hide(axis='columns')


if __name__ == '__main__':
    # Downloading the Dataset
    # Construct the HuggingFace Hub dataset URL
    dataset_url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/{dataset_name}.zip"
    print(f"HuggingFace Dataset URL: {dataset_url}")

    # Set whether to delete the archive file after extracting the dataset
    delete_archive = True

    # Download the dataset if not present
    if dataset_path.is_dir():
        print("Dataset folder already exists")
    else:
        print("Downloading dataset...")
        download_file(dataset_url, archive_dir)

        print("Extracting dataset...")
        file_extract(fname=archive_path, dest=dataset_dir)

        # Delete the archive if specified
        if delete_archive:
            archive_path.unlink()


