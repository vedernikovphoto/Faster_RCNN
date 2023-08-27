from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#########################################################################################################################
#########################################################################################################################

# define GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

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


#########################################################################################################################
#########################################################################################################################

# Load the image
image_path = './Inference/test.jpg'
image = Image.open(image_path)

# Transform the image
image_tensor = to_tensor(image).unsqueeze(0).to(device)  # add batch dimension and move to device

# Perform inference
model.eval()
with torch.no_grad():
    prediction = model(image_tensor)

# Draw boxes on the image
draw = ImageDraw.Draw(image)
for box in prediction[0]['boxes']:
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red')

# Display the image in Jupyter notebook
plt.figure(figsize=(20, 15))  # adjust as needed
plt.imshow(image)
plt.axis('off')  # to hide the axis
plt.show()