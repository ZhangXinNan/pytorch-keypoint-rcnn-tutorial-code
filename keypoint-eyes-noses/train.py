
import math
import json
from tqdm.auto import tqdm
from pathlib import Path
import torch
import torchvision
from contextlib import contextmanager
from cjm_pytorch_utils.core import set_seed, pil_to_tensor, tensor_to_pil, get_torch_device, denorm_img_tensor, move_data_to_device


# Fine-tuning the Model
# Define Utility Functions
# Define a function to create a bounding box that encapsulates the key points
def keypoints_to_bbox(keypoints, offset=10):
    """
    Convert a tensor of keypoint coordinates to a bounding box.

    Args:
    keypoints (Tensor): A tensor of shape (N, 2), where N is the number of keypoints.

    Returns:
    Tensor: A tensor representing the bounding box [xmin, ymin, xmax, ymax].
    """
    x_coordinates, y_coordinates = keypoints[:, 0], keypoints[:, 1]

    xmin = torch.min(x_coordinates)
    ymin = torch.min(y_coordinates)
    xmax = torch.max(x_coordinates)
    ymax = torch.max(y_coordinates)

    bbox = torch.tensor([xmin-offset, ymin-offset, xmax+offset, ymax+offset])

    return bbox



# Define a conditional autocast context manager
@contextmanager
def conditional_autocast(device):
    """
    A context manager for conditional automatic mixed precision (AMP).

    This context manager applies automatic mixed precision for operations if the
    specified device is not a CPU. It's a no-op (does nothing) if the device is a CPU.
    Mixed precision can speed up computations and reduce memory usage on compatible
    hardware, primarily GPUs.

    Parameters:
    device (str): The device type, e.g., 'cuda' or 'cpu', which determines whether
                  autocasting is applied.

    Yields:
    None - This function does not return any value but enables the wrapped code
           block to execute under the specified precision context.
    """

    # Check if the specified device is not a CPU
    if 'cpu' not in device:
        # If the device is not a CPU, enable autocast for the specified device type.
        # Autocast will automatically choose the precision (e.g., float16) for certain
        # operations to improve performance.
        with autocast(device_type=device):
            yield
    else:
        # If the device is a CPU, autocast is not applied.
        # This yields control back to the with-block with no changes.
        yield


# Define the Training Loop
# Function to run a single training/validation epoch
def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.

    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.

    Returns:
        The average loss for the epoch.
    """
    # Set model to training mode
    model.train()

    # Initialize the average loss for the current epoch
    epoch_loss = 0
    # Initialize progress bar with total number of batches in the dataloader
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")

    # Iterate over data batches
    for batch_id, (inputs, targets) in enumerate(dataloader):

        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)
        # Extract the ground truth bounding boxes and labels
        gt_bboxes, gt_labels = zip(*[(d['boxes'].to(device), d['labels'].to(device)) for d in targets])

        # Convert ground truth bounding boxes from 'xyxy' to 'cxcywh' format and only keep center coordinates
        gt_keypoints = torchvision.ops.box_convert(torch.stack(gt_bboxes), 'xyxy', 'cxcywh')[:, :, :2]

        # Initialize a visibility tensor with ones, indicating all keypoints are visible
        visibility = torch.ones(len(inputs), gt_keypoints.shape[1], 1).to(device)
        # Create a visibility mask based on whether the bounding boxes are valid (greater than or equal to 0)
        visibility_mask = (torch.stack(gt_bboxes) >= 0.)[..., 0].view(visibility.shape).to(device)

        # Concatenate the keypoints with the visibility mask, adding a visibility channel to keypoints
        gt_keypoints_with_visibility = torch.concat((
            gt_keypoints,
            visibility * visibility_mask
        ), dim=2)

        # Convert keypoints to bounding boxes for each input and move them to the specified device
        gt_object_bboxes = torch.vstack([keypoints_to_bbox(keypoints) for keypoints in gt_keypoints]).to(device)
        # Initialize ground truth labels as tensor of ones and move them to the specified device
        gt_labels = torch.ones(len(inputs), dtype=torch.int64).to(device)

        # Prepare the targets for the Keypoint R-CNN model
        # This includes bounding boxes, labels, and keypoints with visibility for each input image
        keypoint_rcnn_targets = [
            {'boxes': boxes[None], 'labels': labels[None], 'keypoints': keypoints[None]}
            for boxes, labels, keypoints in zip(gt_object_bboxes, gt_labels, gt_keypoints_with_visibility)
        ]

        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with conditional_autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(keypoint_rcnn_targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(keypoint_rcnn_targets, device))

            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            optimizer.zero_grad()

        loss_item = loss.item()
        epoch_loss += loss_item
        # Update progress bar
        progress_bar.set_postfix(loss=loss_item,
                                 avg_loss=epoch_loss / (batch_id + 1),
                                 lr=lr_scheduler.get_last_lr()[0] if is_training else "")
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    progress_bar.close()
    return epoch_loss / (batch_id + 1)


def train_loop(model,
               train_dataloader,
               valid_dataloader,
               optimizer,
               lr_scheduler,
               device,
               epochs,
               checkpoint_path,
               use_scaler=False):
    """
    Main training loop.

    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device

    Returns:
        None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss

    # Loop over the epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch, is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent /'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()