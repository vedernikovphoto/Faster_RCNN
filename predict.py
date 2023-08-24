import torch
from tqdm import tqdm

# import torchvision
# import json
# import numpy as np
# import os
# import random
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
# import torch.optim as optim
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from PIL import Image
# from torchvision.transforms.functional import to_tensor
# from sklearn.metrics import jaccard_score
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torchvision.transforms import functional as F
# from dataset import *
# from mytransforms import *
# from show_sample import *
# from train_one_epoch import *


# define GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict(model, data_loader, score_threshold=0.8):
    # This function uses the given model to make predictions on the provided data_loader 

    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store the predictions, the images, the ground truth data and the indices.
    predictions = []
    images = []
    ground_truths = []

    # Disable gradient calculation because we are in the testing mode
    with torch.no_grad():
        
        
        # Iterate over each batch of data in the data loader
        for (image, targets) in tqdm(data_loader):
            
            # Move the images to the device (GPU or CPU) that the model is on
            image = list(img.to(device) for img in image)
            
            # Use the model to predict the targets from the images
            prediction = model(image)
            
            # Apply a threshold to the scores
            for pred in prediction:
                confident_indices = pred['scores'] > score_threshold
                pred['boxes'] = pred['boxes'][confident_indices]
                pred['labels'] = pred['labels'][confident_indices]
                pred['scores'] = pred['scores'][confident_indices]
            
            # Move the predictions and images back to CPU memory
            predictions.extend([{k: v.to('cpu') for k, v in pred.items()} for pred in prediction])
            images.extend([img.to('cpu') for img in image])
            
            # Store the ground truth data
            ground_truths.extend(targets)
            
    # Return the predictions, images and ground truths
    return predictions, images, ground_truths