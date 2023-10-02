import torch
import argparse
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from dataset import MyDataset
from mytransforms import MyTransforms
from show_sample import plot_loss


def train_model(num_epochs=2, batch_size=2, lr=0.005, momentum=0.9, weight_decay=0.0005,
                step_size=3, gamma=0.1, root=None, annotations_file=None, model_save_path=None):
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
        root (str, optional): Root path for images. Defaults to None.
        annotations_file (str, optional): Path to annotations file. Defaults to None.
        model_save_path (str, optional): Path to save the trained model weights. Defaults to None.

    Returns:
        list: A list of loss values recorded during training.
    """

    dataset = MyDataset(root=root,
                        annotations_file=annotations_file,
                        transforms=MyTransforms(train=True))

    # Transforms a list of sample tuples into a tuple of lists for batching
    collate_lambda = lambda x: tuple(zip(*x))
    data_loader_train = DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   collate_fn=collate_lambda)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)  # Load a pre-trained model

    num_classes = 3  # 2 classes (person, head) + background

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

    torch.save(model.state_dict(), model_save_path)

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
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train an object detection model.")

    # Adding arguments for the train_model function
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=2, help='Size of the batches during training.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=3, help='Step size for the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for the learning rate scheduler.')
    parser.add_argument('--root', type=str, default='D:\\Object detection\\CrowdHuman_train01\\Images',
                        help='Root path for images.')
    parser.add_argument('--annotations_file', type=str, default='D:\\Object detection\\annotation_train.odgt',
                        help='Path to annotations file.')
    parser.add_argument('--model_save_path', type=str, default='model_weights_final.pth',
                        help='Path to save the trained model weights.')

    args = parser.parse_args()

    # Using the parsed arguments in the train_model function
    loss_values = train_model(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        root=args.root,
        annotations_file=args.annotations_file,
        model_save_path=args.model_save_path
    )

    plot_loss(loss_values)

#     this line has no meaning and I just wanna check if it works right :)
