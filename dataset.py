import torch
import os
import json
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    
    # Initialize dataset
    def __init__(self, root, annotations_file, transforms=None):
        self.root = root  # Path to the images
        self.transforms = transforms  # Image transformations
        
        # Get list of all image files
        self.imgs = list(sorted(os.path.join(root, img_file) for img_file in os.listdir(root)))
        
        # Load annotations from file
        self.annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                self.annotations.append(json.loads(line))  # Parse JSON line and add to list    

        self.labels = []  # Initialize an empty list to store the class labels of objects in each image
        self.boxes = []  # Initialize an empty list to store the bounding boxes of objects in each image
 
        # Adjust the bounding box format from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        def adjust_box_format(box):
            x, y, w, h = box
            return [x, y, x + w, y + h]

        # Iterate over all images and store the annotations in the dataset object
        for img_file in self.imgs:
            
            # Extract the image file name and remove the extension
            image_file_name = os.path.basename(img_file)
            image_file_name_wo_extension = os.path.splitext(image_file_name)[0]

            # Get the corresponding annotation for each image
            annotation = next(ann for ann in self.annotations if ann["ID"] == image_file_name_wo_extension)

            # Get the boxes and labels from the annotation
            gtboxes = annotation["gtboxes"]
            boxes = [adjust_box_format(box["vbox"]) for box in gtboxes if box["tag"] == "person"]
            labels = [1 for _ in boxes]  # Assuming all are "person", thus label is 1

            self.boxes.append(boxes)  # Append the bounding boxes for the current image to the list of all bounding boxes
            self.labels.append(labels)  # Append the labels for the current image to the list of all labels
            

#########################################################################################################################     
#########################################################################################################################     


    # Method to get dataset item at a particular index
    def __getitem__(self, idx):
        
        # Load images and bounding boxes
        img = Image.open(self.imgs[idx]).convert("RGB")  # Open and convert the image at the given index to RGB
        box_list = self.boxes[idx]  # Retrieve the list of bounding boxes for the image at the given index
        
        # Convert the bounding boxes to a PyTorch tensor with float32 data type
        boxes = torch.as_tensor(box_list, dtype=torch.float32)  
        
        # Get the number of objects in the image by counting the number of bounding boxes
        num_objs = len(box_list)  

        # Convert the labels for the image at the given index to a PyTorch tensor with int64 data type
        labels = torch.as_tensor(self.labels[idx], dtype=torch.int64)  
        
        
        # Create a dictionary to store the details of the target object(s) in the current image
        target = {}
        target["boxes"] = boxes  # Store the bounding boxes tensor in the target dictionary under the key "boxes"
        target["labels"] = labels  # Store the labels tensor in the target dictionary under the key "labels"
        target["image_id"] = torch.tensor([idx])  # Store the index of the image in the target dictionary under the key "image_id"

        # Apply transformations if any
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


#########################################################################################################################     
#########################################################################################################################     


    # Method to get length of the dataset
    def __len__(self):
        return len(self.imgs)