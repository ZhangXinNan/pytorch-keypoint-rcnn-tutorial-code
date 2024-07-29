import torch
# Import Keypoint R-CNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import keypointrcnn_resnet50_fpn


def get_model(class_names, device, dtype):
    # Load a pre-trained model
    model = keypointrcnn_resnet50_fpn(weights='DEFAULT')

    # Replace the classifier head with the number of keypoints
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_channels=in_features, num_keypoints=len(class_names))

    # Set the model's device and data type
    model.to(device=device, dtype=dtype)

    # Add attributes to store the device and model name for later reference
    model.device = device
    model.name = 'keypointrcnn_resnet50_fpn'
    return model

