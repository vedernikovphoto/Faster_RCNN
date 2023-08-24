# Step 0: Import necessary libraries and modules
import torch
import torchvision
import json
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.optim as optim

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision.transforms.functional import to_tensor
from sklearn.metrics import jaccard_score
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

from dataset import *
from mytransforms import *
from show_sample import *
from train_one_epoch import *

# Use custom dataset and transformations
dataset = MyDataset(root='D:\\Object detection\\CrowdHuman_train01\\Images',
                    annotations_file='D:\\Object detection\\annotation_train.odgt',
                    transforms=MyTransforms(train=True))


# Define DataLoader for training data
data_loader_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))


# # Display some random samples
# num_samples = 5
# total_samples = 50
# random_indices = random.sample(range(total_samples), num_samples)

# for i in random_indices:
#     sample = dataset[i]
#     print(f"Sample #{i}:")
#     # print(f"Image shape: {sample[0].size}")
#     # print(f"Targets: {sample[1]}")
#     show_sample(sample)


#########################################################################################################################
#########################################################################################################################


# Fine-tune the model

# Define the device to use for computation
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Define the number of classes
num_classes = 2  # 1 class (person) + background

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the right device
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Define a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

loss_values = []  # List to store loss values over epochs

num_epochs = 30  # Number of epochs for training

# Train the model for num_epochs epochs
for epoch in range(num_epochs):
    epoch_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=4)
    loss_values.append(epoch_loss)  # Append the loss value to the list
    lr_scheduler.step()  # Update the learning rate
    torch.cuda.empty_cache()  # Clean up any unneeded CUDA memory

# Save the trained model weights
torch.save(model.state_dict(), 'model_weights_final.pth')




# Plot the training loss over epochs
plt.figure(figsize=(10,5), dpi=300)
plt.title("Training Loss per Epoch")
plt.plot(loss_values)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.tick_params(direction='in')
plt.grid()
plt.show()