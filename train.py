import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from dataset import *
from mytransforms import *
from show_sample import *
from train_one_epoch import *


def train_model(num_epochs=2, batch_size=2, lr=0.005, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):

    # Use custom dataset and transformations
    dataset = MyDataset(root='D:\\Object detection\\CrowdHuman_train01\\Images',
                        annotations_file='D:\\Object detection\\annotation_train.odgt',
                        transforms=MyTransforms(train=True))

    # Define DataLoader for training data
    data_loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Define the device to use for computation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load a pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

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
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Define a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_values = []  # List to store loss values over epochs

    # Train the model for num_epochs epochs
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=4)
        loss_values.append(epoch_loss)  # Append the loss value to the list
        lr_scheduler.step()  # Update the learning rate
        torch.cuda.empty_cache()  # Clean up any unneeded CUDA memory

    # Save the trained model weights
    torch.save(model.state_dict(), 'model_weights_final.pth')

    return loss_values


if __name__ == "__main__":
    loss_values = train_model()
    plot_loss(loss_values)