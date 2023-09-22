import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def show_sample(sample):

    # Unpack the sample into the image and target dictionary
    img, target = sample

    # Change order of the axes from (C, H, W) to (H, W, C) because matplotlib requires the channel dimension to be last
    img = img.permute(1, 2, 0)

    fig, ax = plt.subplots(1)
    ax.imshow(np.array(img))

    for box in target['boxes']:

        # Unpack the bounding box coordinates
        xmin, ymin, xmax, ymax = box

        # Create a rectangle patch representing the bounding box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def plot_image_with_boxes(image, gt_boxes, pred_boxes):

    fig, ax = plt.subplots(1, dpi=96)
    ax.imshow(image)

    # Draw ground truth bounding boxes in red
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    # Draw predicted bounding boxes in blue
    for box in pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def plot_image_with_all_boxes(image, gt_boxes, pred_boxes, raw_pred_boxes):

    fig, ax = plt.subplots(1, dpi=96)
    ax.imshow(image)

    # Plot ground truth boxes in red
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot boxes predicted by fine-tuned model in blue
    for box in pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot boxes predicted by ResNet pre-trained model in green
    for box in raw_pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()


def plot_loss(loss_values):

    plt.figure(figsize=(10, 5), dpi=300)
    plt.title("Training Loss per Epoch")
    plt.plot(loss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tick_params(direction='in')
    plt.grid()
    plt.show()


def plot_predictions(test_images, test_ground_truths, test_predictions):

    num_test_images = len(test_images)
    random_index = random.randint(0, num_test_images-1)  # Randomly select an image
    test_image = test_images[random_index].permute(1, 2, 0).cpu().numpy()  # Select a prediction to plot

    # Get corresponding ground truth and prediction boxes
    gt_boxes = test_ground_truths[random_index]['boxes'].cpu().numpy()
    pred_boxes = test_predictions[random_index]['boxes'].cpu().numpy()

    print('\nPlotting the predictions of fine-tuned model and ground truth bounding boxes: ')
    print("\033[94m" + "Predicted bounding boxes are in blue" + "\033[0m")
    print("\033[91m" + "Ground truth bounding boxes are in red" + "\033[0m")

    plot_image_with_boxes(test_image, gt_boxes, pred_boxes)


def plot_predictions_total(test_images, test_ground_truths, test_predictions, test_predictions_raw):

    num_test_images = len(test_images)
    random_index = random.randint(0, num_test_images-1)
    test_image = test_images[random_index].permute(1, 2, 0).cpu().numpy()

    # Get corresponding ground truth and prediction boxes
    gt_boxes = test_ground_truths[random_index]['boxes'].cpu().numpy()
    pred_boxes = test_predictions[random_index]['boxes'].cpu().numpy()
    raw_pred_boxes = test_predictions_raw[random_index]['boxes'].cpu().numpy()

    print('\nPlotting the predictions of fine-tuned and ResNet models as well as ground truth bounding boxes: ')
    print("\033[91m" + "Ground truth bounding boxes are in red" + "\033[0m")
    print("\033[92m" + "Predicted bounding boxes from fine-tuned model are in green" + "\033[0m")
    print("\033[94m" + "Pre-trained bounding boxes from ResNet model are in blue" + "\033[0m")

    plot_image_with_all_boxes(test_image, gt_boxes, pred_boxes, raw_pred_boxes)
