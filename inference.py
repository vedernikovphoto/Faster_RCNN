from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def init_resnet_50():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = fasterrcnn_resnet50_fpn(weights=None)  # Load a pre-trained model

    # Replace the classifier with a new one, that has user-defined number of classes
    num_classes = 2  # 1 class (person) + background

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    model.load_state_dict(torch.load('model_weights_final.pth', map_location=device))

    return model, device


def model_inference(model, device, image_path):

    image = Image.open(image_path)

    image_tensor = to_tensor(image).unsqueeze(0).to(device)  # Transform the image

    # Perform inference
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    # Draw boxes on the image
    draw = ImageDraw.Draw(image)
    for box in prediction[0]['boxes']:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red')

    plt.figure(figsize=(20, 15))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    image_path = './Inference/test.jpg'
    resnet_50, device = init_resnet_50()
    model_inference(model=resnet_50,
                    device=device,
                    image_path=image_path)
