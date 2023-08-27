# import torch
import torchvision
# import json
import numpy as np
# import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# import torch.optim as optim

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from PIL import Image
# from torchvision.transforms.functional import to_tensor
# from sklearn.metrics import jaccard_score
# from torchvision import transforms
from torch.utils.data import DataLoader
# from torchvision.transforms import functional as F
from tqdm import tqdm

from dataset import *
from mytransforms import *
from show_sample import *
# from train_one_epoch import *
from predict import *
from test_one_epoch import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create a Dataset for your test data
test_dataset = MyDataset(root='D:\\Object detection\\CrowdHuman_val\\Images', 
                           annotations_file='D:\\Object detection\\annotation_val.odgt',
                           transforms=MyTransforms(train=False))

# Create a DataLoader for your test data
data_loader_test = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))


# Display some random test samples
# Select a random subset of the test data to display
# num_samples = 5
# total_samples = 20
# random_indices = random.sample(range(total_samples), num_samples)

# # Print some samples
# for i in random_indices:  # adjust this to see more/fewer images
#     sample = test_dataset[i]
#     print(f"Sample #{i}:")
# #     print(f"Image shape: {sample[0].size}")
# #     print(f"Targets: {sample[1]}")
#     show_sample(sample)


#########################################################################################################################
#########################################################################################################################

# Load a pre-trained model without pretrained weights
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Load your pretrained ResNet weights from a local path
pretrained_weights_path = 'fasterrcnn_resnet50_fpn_coco.pth'
model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

# Replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (person) + background

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the right device
model.to(device)

# Load your trained model
model.load_state_dict(torch.load('model_weights_final.pth', map_location=device))

# Make sure the model is in evaluation mode
model.eval()

torch.manual_seed(0)
np.random.seed(0)

# Use the model to make predictions on the test data
print('\nUse the fine-tuned model to make predictions on the test data: ')
test_predictions, test_images, test_ground_truths = predict(model, data_loader_test)


#########################################################################################################################
#########################################################################################################################


# Get number of test images
num_test_images = len(test_images)

# Randomly select an image
random_index = random.randint(0, num_test_images-1)

# Select a prediction to plot
test_image = test_images[random_index].permute(1, 2, 0).cpu().numpy()

# Get corresponding ground truth and prediction boxes
gt_boxes = test_ground_truths[random_index]['boxes'].cpu().numpy()
pred_boxes = test_predictions[random_index]['boxes'].cpu().numpy()

print('\nPlotting the predictions of fine-tuned model and ground truth bounding boxes: ')
print("\033[94m" + "Predicted bounding boxes are in blue" + "\033[0m")
print("\033[91m" + "Ground truth bounding boxes are in red" + "\033[0m")

# Plot the selected image with both ground truth and predicted bounding boxes
plot_image_with_boxes(test_image, gt_boxes, pred_boxes)


#########################################################################################################################
#########################################################################################################################

print('\nCalculating the performance of fine-tuned model: ')

# Calculate the average IoU on the test data using fune-tuned model
test_iou = test_one_epoch(model, data_loader_test, device)
print(f'\nTest IoU of fine-tuned model: {test_iou}')


#########################################################################################################################
#########################################################################################################################

print('\nCalculating the performance of ResNet model and plot the predictions: ')

# Calculate the performance of ResNet model
# Load the pretrained model
model_raw = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Move the model to the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_raw.to(device)

# Compute test IoU
test_iou = test_one_epoch(model_raw, data_loader_test, device)
print(f'\nTest IoU of ResNet model: {test_iou}')


# Set the model to evaluation mode
model_raw.eval()

torch.manual_seed(0)
np.random.seed(0)

# Use the model to generate predictions for the test data set
test_predictions_raw, test_images_raw, test_ground_truths_raw = predict(model_raw, data_loader_test)

# Select a random index from the range of number of test images
random_index = random.randint(0, len(test_images_raw)-1)

# Using the random index, select a corresponding image from test_images_raw and reorder its dimensions from (C, H, W) to (H, W, C), and convert the tensor to numpy array for displaying it
test_image_raw = test_images_raw[random_index].permute(1, 2, 0).cpu().numpy()

# Get the predicted bounding boxes for the randomly selected image by using the same random index on test_predictions_raw
test_prediction_boxes_raw = test_predictions_raw[random_index]['boxes'].cpu().numpy()

# Use the helper function plot_image_with_boxes to display the image with the predicted bounding boxes. We pass an empty list for the ground truth boxes because we only want to plot predicted boxes.
plot_image_with_boxes(test_image_raw, [], test_prediction_boxes_raw)  # passing empty list for ground truth boxes


#########################################################################################################################
#########################################################################################################################


print('\nPlotting the predictions of fine-tuned and pre-trained models as well as ground truth bounding boxes: ')

# Randomly select an image
random_index = random.randint(0, num_test_images-1)
# random_index=30

# Select a prediction to plot
test_image = test_images[random_index].permute(1, 2, 0).cpu().numpy()

# Get corresponding ground truth and prediction boxes
gt_boxes = test_ground_truths[random_index]['boxes'].cpu().numpy()
pred_boxes = test_predictions[random_index]['boxes'].cpu().numpy()
raw_pred_boxes = test_predictions_raw[random_index]['boxes'].cpu().numpy()

print("\033[91m" + "Ground truth bounding boxes are in red" + "\033[0m")
print("\033[92m" + "Predicted bounding boxes are in green" + "\033[0m")
print("\033[94m" + "Pre-trained bounding boxes are in blue" + "\033[0m")

plot_image_with_all_boxes(test_image, gt_boxes, pred_boxes, raw_pred_boxes)
