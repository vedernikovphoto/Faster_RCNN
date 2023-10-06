import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import ImageFont, ImageDraw


def plot_image_with_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_labels):
    fig, ax = plt.subplots(1, dpi=96)
    ax.imshow(image)
    colors = {'human': 'blue', 'head': 'red'}

    # Draw ground truth bounding boxes
    for box, label in zip(gt_boxes, gt_labels):
        color = colors.get(label)
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    # Draw predicted bounding boxes
    for box, label in zip(pred_boxes, pred_labels):
        color = colors.get(label)
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, linestyle='--', edgecolor=color, facecolor='none')
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
    test_image = test_images[random_index].permute(1, 2, 0).cpu().numpy()  # Select image to plot

    # Get corresponding ground truth and prediction boxes and labels
    gt_boxes = test_ground_truths[random_index]['boxes'].cpu().numpy()
    gt_labels = test_ground_truths[random_index]['labels'].cpu().numpy()
    gt_labels = ['human' if l == 1 else 'head' if l == 2 else 'background' for l in gt_labels]

    pred_boxes = test_predictions[random_index]['boxes'].cpu().numpy()
    pred_labels = test_predictions[random_index]['labels'].cpu().numpy()
    pred_labels = ['human' if l == 1 else 'head' if l == 2 else 'background' for l in pred_labels]

    print('\nPlotting the predictions of model and ground truth bounding boxes: ')
    print("\033[94m" + "-- Human bounding boxes are in blue" + "\033[0m")
    print("\033[91m" + "-- Head bounding boxes are in red" + "\033[0m")
    print("-- Bounding boxes of the fine-tuned model are dashed lines.")
    print("-- Ground truth bounding boxes are solid lines.")

    plot_image_with_boxes(test_image, gt_boxes, gt_labels, pred_boxes, pred_labels)


def plot_inference(image, prediction, score_threshold):
    # Draw boxes on the image
    draw = ImageDraw.Draw(image)
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    for box, label, score in zip(prediction[0]['boxes'], labels, scores):
        if score < score_threshold:  # Skip predictions with low confidence
            continue
        if label == 1:  # person
            color = 'blue'
        elif label == 2:  # head
            color = 'red'

        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
        draw.text((box[0], box[1]),
                  text=f'{score:.2f}',
                  fill=color,
                  font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24))

    plt.figure(figsize=(20, 15))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
