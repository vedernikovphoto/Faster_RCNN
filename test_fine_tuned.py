import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from dataset import *
from mytransforms import *
from show_sample import *
from predict import *
from test_one_epoch import *


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
    model = get_model(num_classes, pretrained_weights_path, device)
    model.load_state_dict(torch.load('model_weights_final.pth', map_location=device))
    model.eval()

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Use the model to make predictions on the test data
    print('\nUse the fine-tuned model to make predictions on the test data: ')
    test_predictions, test_images, test_ground_truths = predict(model, data_loader_test)
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
    test_predictions_raw, test_images_raw, test_ground_truths_raw = predict(model_raw, data_loader_test)
    calculate_performance(model_raw, data_loader_test, device, "ResNet model")
    plot_predictions(test_images_raw, test_ground_truths_raw, test_predictions_raw)

    # Plot the predictions from fine-tuned mode, ResNet model and ground truth
    plot_predictions_total(test_images, test_ground_truths, test_predictions, test_predictions_raw)


def calculate_performance(model, data_loader_test, device, model_description):
    """Calculate the performance of a given model on the test data loader."""
    print(f'\nCalculating the performance of {model_description}: ')

    # Calculate the average IoU on the test data using the provided model
    test_iou = test_one_epoch(model, data_loader_test, device)
    print(f'\nTest IoU of {model_description}: {test_iou}')


if __name__ == '__main__':
    main()
