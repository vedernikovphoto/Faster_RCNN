from PIL import Image, ImageDraw
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def init_resnet_50(model_weights_final=None):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = fasterrcnn_resnet50_fpn(weights=None)  # Initialize the model without pre-trained weights

    # Replace the classifier with a new one that has a user-defined number of classes
    num_classes = 2  # 1 class (person) + background

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    model.load_state_dict(torch.load(model_weights_final, map_location=device))

    return model, device


def model_inference(model, device, image_path):

    image = Image.open(image_path)

    image_tensor = to_tensor(image).unsqueeze(0).to(device)  # Convert the image to a tensor

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
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Perform inference using an object detection model.")

    # Adding arguments for the model_inference function
    parser.add_argument('--image_path', type=str, default='../Inference/test.jpg',
                        help='Path to the image on which to perform inference.')
    parser.add_argument('--model_weights_final', type=str, default='model_weights_final.pth',
                        help='Name of the .pth file containing fine-tuned weights.')

    args = parser.parse_args()

    # Using the parsed arguments in the model_inference function
    resnet_50, device = init_resnet_50(model_weights_final=args.model_weights_final)
    model_inference(model=resnet_50,
                    device=device,
                    image_path=args.image_path)
