import torch
import argparse

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from show_sample import plot_inference


def init_resnet_50(model_weights_final=None):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = fasterrcnn_resnet50_fpn(weights=None)  # Initialize the model without pre-trained weights

    # Replace the classifier with a new one that has a user-defined number of classes
    num_classes = 3  # 2 classes (person, head) + background

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    model.load_state_dict(torch.load(model_weights_final, map_location=device))

    return model, device


def model_inference(model, device, image_path, score_threshold):
    image = Image.open(image_path)
    image_tensor = to_tensor(image).unsqueeze(0).to(device)  # Convert the image to a tensor

    # Perform inference
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    plot_inference(image, prediction, score_threshold)


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Perform inference using an object detection model.")

    # Adding arguments for the model_inference function
    parser.add_argument('--image_path', type=str, default='../Inference/test.jpg',
                        help='Path to the image on which to perform inference.')
    parser.add_argument('--model_weights_final', type=str, default='model_weights_final.pth',
                        help='Name of the .pth file containing fine-tuned weights.')
    parser.add_argument('--score_threshold', type=float, default=0.8,
                        help='Confidence threshold.')

    args = parser.parse_args()

    # Using the parsed arguments in the model_inference function
    resnet_50, device = init_resnet_50(model_weights_final=args.model_weights_final)
    model_inference(model=resnet_50,
                    device=device,
                    image_path=args.image_path,
                    score_threshold=args.score_threshold)
