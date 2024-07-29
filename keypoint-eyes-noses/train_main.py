
import os
import datetime
import multiprocessing
from pathlib import Path
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# Import tqdm for progress bar
from tqdm.auto import tqdm

# Import utility functions
from cjm_pandas_utils.core import markdown_to_pandas
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from cjm_psl_utils.core import download_file, file_extract
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop #, RandomPixelCopy
from cjm_pytorch_utils.core import set_seed, pil_to_tensor, tensor_to_pil, get_torch_device, denorm_img_tensor, move_data_to_device

from labelme_keypoint_dataset import LabelMeKeypointDataset, tuple_batch
from data_aug import *
from model import *
from setting_up import project_dir
from load_explore_dataset import *
from train import train_loop


def main():

    # Setting Up the Project
    # Setting a Random Number Seed
    # Set the seed for generating random numbers in PyTorch, NumPy, and Python's random module.
    seed = 123
    set_seed(seed)
    # Setting the Device and Data Type
    if torch.cuda.is_available():
        device = get_torch_device()
    else:
        device = 'cpu'
    dtype = torch.float32
    print(device, dtype)

    class_names = ['left-eye', 'right-eye', 'nose']
    # Initialize Dataset
    # Create a mapping from class names to class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(class_names, class_to_idx)

    # Preparing the Data
    # Training-Validation Split
    # Get the list of image IDs
    img_keys = list(img_dict.keys())

    # Shuffle the image IDs
    random.shuffle(img_keys)

    # Define the percentage of the images that should be used for training
    train_pct = 0.9
    val_pct = 0.1

    # Calculate the index at which to split the subset of image paths into training and validation sets
    train_split = int(len(img_keys) * train_pct)
    val_split = int(len(img_keys) * (train_pct + val_pct))

    # Split the subset of image paths into training and validation sets
    train_keys = img_keys[:train_split]
    val_keys = img_keys[train_split:]

    # Print the number of images in the training and validation sets
    pd.Series({
        "Training Samples:": len(train_keys),
        "Validation Samples:": len(val_keys)
    }).to_frame().style.hide(axis='columns')

    # Instantiate the dataset using the defined transformations
    train_dataset = LabelMeKeypointDataset(train_keys, annotation_df, img_dict, class_to_idx, train_tfms)
    valid_dataset = LabelMeKeypointDataset(val_keys, annotation_df, img_dict, class_to_idx, valid_tfms)

    # Print the number of samples in the training and validation datasets
    pd.Series({
        'Training dataset size:': len(train_dataset),
        'Validation dataset size:': len(valid_dataset)}
    ).to_frame().style.hide(axis='columns')

    # Initialize DataLoaders
    # Set the training batch size
    bs = 4

    # Set the number of worker processes for loading data. This should be the number of CPUs available.
    # num_workers = multiprocessing.cpu_count()
    num_workers = 1

    # Define parameters for DataLoader
    data_loader_params = {
        'batch_size': bs,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,
        # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in device,
        # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        'pin_memory_device': device if 'cuda' in device else '',
        # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': tuple_batch,
    }

    # Create DataLoader for training data. Data is shuffled for every epoch.
    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)

    # Create DataLoader for validation data. Shuffling is not necessary for validation data.
    valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

    # Print the number of batches in the training and validation DataLoaders
    print(f'Number of batches in train DataLoader: {len(train_dataloader)}')
    print(f'Number of batches in validation DataLoader: {len(valid_dataloader)}')

    model = get_model(class_names, device, dtype)
    # Set the Model Checkpoint Path
    # Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a directory to store the checkpoints if it does not already exist
    checkpoint_dir = Path(project_dir/f"{timestamp}")

    # Create the checkpoint directory if it does not already exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # The model checkpoint path
    checkpoint_path = checkpoint_dir/f"{model.name}.pth"

    print(checkpoint_path)

    # Configure the Training Parameters
    # Learning rate for the model
    lr = 5e-4

    # Number of training epochs
    epochs = 70

    # AdamW optimizer; includes weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Learning rate scheduler; adjusts the learning rate during training
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=lr,
                                                       total_steps=epochs*len(train_dataloader))

    # Train the Model
    train_loop(model=model,
               train_dataloader=train_dataloader,
               valid_dataloader=valid_dataloader,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               device=torch.device(device),
               epochs=epochs,
               checkpoint_path=checkpoint_path,
               use_scaler=True)


if __name__ == '__main__':
    main()