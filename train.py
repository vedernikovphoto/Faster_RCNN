import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from dataset import MyDataset
from mytransforms import MyTransforms
from show_sample import plot_loss


def train_model(num_epochs=2, batch_size=2, lr=0.005, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):
    """
    Trains the model using the specified parameters and returns the loss values recorded during training.

    Args:
        num_epochs (int): Number of epochs for training.
        batch_size (int): Size of the batches during training.
        lr (float): Learning rate.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Gamma for the learning rate scheduler.

    Returns:
        list: A list of loss values recorded during training.
    """

    dataset = MyDataset(root='D:\\Object detection\\CrowdHuman_train01\\Images',
                        annotations_file='D:\\Object detection\\annotation_train.odgt',
                        transforms=MyTransforms(train=True))

    data_loader_train = DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)  # Load a pre-trained model

    num_classes = 2  # 1 class (person) + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features  # Number of input features for the classifier

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Construct an optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_values = []

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=4)
        loss_values.append(epoch_loss)
        lr_scheduler.step()  # Update the learning rate
        torch.cuda.empty_cache()  # Clean up any unneeded CUDA memory

    torch.save(model.state_dict(), 'model_weights_final.pth')

    return loss_values


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    Trains the model for one epoch and returns the average loss.

    Args:
        model: The model to be trained.
        optimizer: The optimizer for training the model.
        data_loader: DataLoader providing the training data.
        device: The device on which to perform the computation (CPU or GPU).
        epoch: The current epoch number.
        print_freq: Frequency at which to print the loss values.

    Returns:
        float: The average loss over the epoch.
    """

    model.train()
    losses = []

    for step, (images, targets) in enumerate(data_loader):
        # Move images and targets to the specified device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses_dict = {k: v.item() for k, v in loss_dict.items()}  # Convert the loss tensor to Python float
        loss_value = sum(losses_dict.values())
        losses.append(loss_value)

        if step % print_freq == 0:
            print(f"Epoch: {epoch}, Iteration: {step}, Loss: {loss_value}")

        optimizer.zero_grad()
        sum(loss_dict.values()).backward()
        optimizer.step()

    return sum(losses) / len(losses)


if __name__ == "__main__":
    loss_values = train_model()
    plot_loss(loss_values)
