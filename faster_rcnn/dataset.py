import torch
import os
import json
from PIL import Image


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotations_file, transforms=None):

        self.root = root
        self.transforms = transforms

        # Get list of all image files
        self.imgs = list(sorted(os.path.join(root, img_file) for img_file in os.listdir(root)))

        self.annotations = []  # List to store the annotations for each image in the dataset

        with open(annotations_file, 'r') as f:
            for line in f:
                self.annotations.append(json.loads(line))  # Parse JSON line and add to list    

        self.labels = []  # List to store the class labels of objects in each image
        self.boxes = []  # List to store the bounding boxes of objects in each image

        # Adjust the bounding box format from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        def adjust_box_format(box):
            x, y, w, h = box
            return [x, y, x + w, y + h]

        # Iterate over all images and store the annotations in the dataset object
        for img_file in self.imgs:

            image_file_name = os.path.basename(img_file)  # Name without the preceding directory path
            image_file_name_wo_extension = os.path.splitext(image_file_name)[0]

            # Get the corresponding annotation for each image
            annotation = next(ann for ann in self.annotations if ann["ID"] == image_file_name_wo_extension)

            # Get the boxes and labels from the annotation
            gtboxes = annotation["gtboxes"]

            boxes_person = [adjust_box_format(box["vbox"]) for box in gtboxes if box["tag"] == "person"]
            labels_person = [1 for _ in boxes_person]
            boxes_head = [adjust_box_format(box["hbox"]) for box in gtboxes if box["tag"] == "person"]
            labels_head = [2 for _ in boxes_head]

            boxes = boxes_person + boxes_head
            labels = labels_person + labels_head

            self.boxes.append(boxes)  # Append the bounding boxes for current image to the list of all bounding boxes
            self.labels.append(labels)  # Append the labels for the current image to the list of all labels

    def __getitem__(self, idx):

        # Load images and bounding boxes
        img = Image.open(self.imgs[idx]).convert("RGB")  # Open and convert the image at the given index to RGB
        box_list = self.boxes[idx]  # Retrieve the list of bounding boxes for the image at the given index

        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        labels = torch.as_tensor(self.labels[idx], dtype=torch.int64)

        # Create a dictionary to store the details of the target object(s) in the current image
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
