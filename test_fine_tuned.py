import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from dataset import MyDataset
from mytransforms import MyTransforms
from show_sample import plot_predictions, plot_predictions_total
from predict import predict


def get_model(num_classes, pretrained_weights_path, device):
    """
    Returns a Faster R-CNN model with a ResNet-50 backbone and FPN. The model is initialized with custom pre-trained
    weights and the number of output classes is set to num_classes.

    Args:
    num_classes (int): Number of classes for classification.
    pretrained_weights_path (str): Path to the pre-trained weights.
    device (torch.device): Device to load the model onto.

    Returns:
    model (torch.nn.Module): The configured model.
    """

    model = fasterrcnn_resnet50_fpn(weights=None)  # Load a pre-trained model without pretrained weights
    model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))  # Load pretrained ResNet weights

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    return model


def main():
    """Main function to execute the model training and prediction process."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create a Dataset and Dataloader for test data
    test_dataset = MyDataset(root='D:\\Object detection\\CrowdHuman_val\\Images',
                             annotations_file='D:\\Object detection\\annotation_val.odgt',
                             transforms=MyTransforms(train=False))

    data_loader_test = DataLoader(test_dataset, batch_size=2,
                                  shuffle=True, num_workers=0,
                                  collate_fn=lambda x: tuple(zip(*x)))

    # Number of classes and pre-trained weights path
    num_classes = 2  # 1 class (person) + background
    pretrained_weights_path = 'fasterrcnn_resnet50_fpn_coco.pth'

    # Initialize and setup model
    model = get_model(num_classes=num_classes,
                      pretrained_weights_path=pretrained_weights_path,
                      device=device)
    model.load_state_dict(torch.load('model_weights_final.pth', map_location=device))
    model.eval()

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Use the model to make predictions on the test data
    print('\nUse the fine-tuned model to make predictions on the test data: ')
    test_predictions, test_images, test_ground_truths = predict(model=model,
                                                                data_loader=data_loader_test,
                                                                score_threshold=0.8)
    calculate_performance(model, data_loader_test, device, "fine-tuned model")
    plot_predictions(test_images, test_ground_truths, test_predictions)

    # Load the ResNet pretrained model
    model_raw = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model_raw.to(device)
    model_raw.eval()

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Use the model to make predictions on the test data
    print('\nUse the ResNet model to make predictions on the test data: ')
    test_predictions_raw, test_images_raw, test_ground_truths_raw = predict(model=model_raw,
                                                                            data_loader=data_loader_test,
                                                                            score_threshold=0.8)
    calculate_performance(model_raw, data_loader_test, device, "ResNet model")
    plot_predictions(test_images_raw, test_ground_truths_raw, test_predictions_raw)

    # Plot the predictions from fine-tuned mode, ResNet model and ground truth
    plot_predictions_total(test_images, test_ground_truths, test_predictions, test_predictions_raw)


def calculate_performance(model, data_loader_test, device, model_description):
    """Calculate the performance of a given model on the test data loader."""
    print(f'\nCalculating the performance of {model_description}: ')

    # Calculate the average IoU on the test data using the provided model
    test_iou = test_one_epoch(model=model,
                              data_loader=data_loader_test,
                              device=device)
    print(f'\nTest IoU of {model_description}: {test_iou}')


def test_one_epoch(model, data_loader, device):

    model.eval()

    ious = []  # IoUs of each prediction with the ground truth

    for images, targets in tqdm(data_loader, desc="Testing"):

        # Move the images and targets to the device (GPU or CPU)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Disable gradient calculation since we run inference
        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, targets):

            # Get the predicted boxes and ground truth boxes
            pred_boxes = output['boxes'].cpu().numpy()
            gt_boxes = target['boxes'].cpu().numpy()

            for pred_box in pred_boxes:
                iou_scores = []

                for gt_box in gt_boxes:

                    # Calculate the IoU of the predicted box and the ground truth box
                    iou = calculate_iou(pred_box, gt_box)
                    iou_scores.append(iou)

                max_iou_score = max(iou_scores)  # Get the maximum IoU score for the predicted box
                ious.append(max_iou_score)

    return sum(ious) / len(ious)


def calculate_iou(pred_box, gt_box):

    # Determine the coordinates of the intersection rectangle
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    # Calculate the area of intersection rectangle
    width = (x2 - x1)
    height = (y2 - y1)

    # if there is no overlap, return 0
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # Calculate the combined area
    area_a = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_b = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + 1e-6)  # Calculate the IoU

    return iou


if __name__ == '__main__':
    main()
