


# Making Predictions with the Model
# Prepare Input Data
# Choose a random item from the validation set
file_id = val_keys[0]

# Retrieve the image file path associated with the file ID
test_file = img_dict[file_id]

# Open the test file
test_img = Image.open(test_file).convert('RGB')

input_img = resize_img(test_img, target_sz=train_sz, divisor=1)

# Calculate the scale between the source image and the resized image
min_img_scale = min(test_img.size) / min(input_img.size)

display(test_img)

# Print the prediction data as a Pandas DataFrame for easy formatting
pd.Series({
    "Source Image Size:": test_img.size,
    "Input Dims:": input_img.size,
    "Min Image Scale:": min_img_scale,
    "Input Image Size:": input_img.size
}).to_frame().style.hide(axis='columns')

# Get Target Annotation Data
# Extract the source annotations for the test image
gt_labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
gt_keypoints = torch.tensor(np.array([shape['points'] for shape in annotation_df.loc[file_id]['shapes']])).reshape(-1,2)
gt_keypoints_bboxes = torch.cat((gt_keypoints, torch.ones(len(gt_keypoints), 2)*BBOX_DIM), dim=1)

# Pass Input Data to the Model
# Set the model to evaluation mode
model.eval();

# Ensure the model and input data are on the same device
model.to(device);
input_tensor = transforms.Compose([transforms.ToImage(),
                                   transforms.ToDtype(torch.float32, scale=True)])(input_img)[None].to(device)

# Make a prediction with the model
with torch.no_grad():
    model_output = model(input_tensor)[0]

# Filter the Model Output
# Set the confidence threshold
conf_threshold = 0.8

# Filter the output based on the confidence threshold
scores_mask = model_output['scores'] > conf_threshold

# Extract and scale the predicted keypoints
predicted_keypoints = (model_output['keypoints'][scores_mask])[:,:,:-1].reshape(-1,2)*min_img_scale


# Compare Model Predictions with the Source Annotations
# Annotate the test image with the ground-truth annotations
gt_annotated_tensor = draw_bboxes(
    image=transforms.PILToTensor()(test_img),
    boxes=torchvision.ops.box_convert(torch.Tensor(gt_keypoints_bboxes), 'cxcywh', 'xyxy'),
    # labels=gt_labels,
    colors=[int_colors[i] for i in [class_names.index(label) for label in gt_labels]]
)

# Prepare the labels and bounding box annotations for the test image
labels = class_names*sum(scores_mask).item()
keypoints_bboxes = torch.cat((predicted_keypoints.cpu(), torch.ones(len(predicted_keypoints), 2)), dim=1)

# Annotate the test image with the model predictions
annotated_tensor = draw_bboxes(
    image=transforms.PILToTensor()(test_img),
    boxes=torchvision.ops.box_convert(torch.Tensor(keypoints_bboxes), 'cxcywh', 'xyxy'),
    # labels=labels,
    colors=[int_colors[i] for i in [class_names.index(label) for label in labels]]
)

stack_imgs([tensor_to_pil(gt_annotated_tensor), tensor_to_pil(annotated_tensor)])
