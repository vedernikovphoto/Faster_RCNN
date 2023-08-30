import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def show_sample(sample):
    """
    Plots sample from dataset
    """

    # Unpack the sample into the image and target dictionary
    img, target = sample

    # Change the order of the axes from (C, H, W) to (H, W, C) because matplotlib requires the channel dimension to be last
    img = img.permute(1, 2, 0)

    fig, ax = plt.subplots(1) # Create a new figure and a set of subplots, in this case only one
    ax.imshow(np.array(img))  # Display the image on the axes

    # Iterate over the bounding boxes in the target
    for box in target['boxes']:

        # Unpack the bounding box coordinates
        xmin, ymin, xmax, ymax = box

        # Create a rectangle patch representing the bounding box
        rect = patches.Rectangle((xmin, ymin), xmax -xmin, ymax -ymin, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle patch to the axes
        ax.add_patch(rect)
    plt.show()


def plot_image_with_boxes(image, gt_boxes, pred_boxes):
    """
    Plots an image with both ground truth and predicted bounding boxes
    """

    fig, ax = plt.subplots(1, dpi=96)
    # Display the image
    ax.imshow(image)
    
    # Draw ground truth bounding boxes in red
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Draw predicted bounding boxes in blue
    for box in pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def plot_image_with_all_boxes(image, gt_boxes, pred_boxes, raw_pred_boxes):
    """
    Plots image with all three types of bounding boxes
    """

    fig, ax = plt.subplots(1, dpi=96)
    ax.imshow(image)

    # Plot ground truth boxes in red
    for box in gt_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot boxes predicted by your model in blue
    for box in pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot boxes predicted by the pre-trained model in green
    for box in raw_pred_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()